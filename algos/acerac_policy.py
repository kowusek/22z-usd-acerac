"""Policies: abstract base class and concrete implementations."""

import collections
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import math

import gym
import numpy as np
import torch as th
from torch import nn

from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.preprocessing import (
    get_flattened_obs_dim,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.policies import BasePolicy


class AceracPolicy(BasePolicy):
    """
    Policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param alpha: Action noise coefficient from the paper.
    """

    actor: nn.Module
    critic: nn.Module

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Union[Schedule, Tuple[Schedule, Schedule]],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        log_std_init: float = 0.0,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        alpha: float = 0.5,
        noise_c: float = 0.3
    ):

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        super().__init__(
            observation_space,
            action_space,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

        # Default network architecture, from stable-baselines
        if net_arch is None:
            net_arch = [dict(pi=[64, 64], vf=[64, 64])]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        self.log_std_init = log_std_init
        dist_kwargs = None

        self.dist_kwargs = dist_kwargs

        # Action distribution
        self.action_dist = make_proba_distribution(
            action_space, dist_kwargs=dist_kwargs
        )
        self.noise_c = th.diag(th.ones(self.action_dist.action_dim) * noise_c)
        self.noise_dist = th.distributions.MultivariateNormal(th.zeros(self.action_dist.action_dim), self.noise_c)
        self.action_dist.requires_grad = False
        self.log_std = th.ones(self.action_dist.action_dim) * log_std_init
        self.alpha = alpha
        self.alpha_sqrt = math.sqrt(1 - alpha**2)

        self._build(lr_schedule)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        default_none_kwargs = self.dist_kwargs or collections.defaultdict(lambda: None)

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                squash_output=default_none_kwargs["squash_output"],
                full_std=default_none_kwargs["full_std"],
                use_expln=default_none_kwargs["use_expln"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                ortho_init=self.ortho_init,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def reset_noise(self, n_envs: int = 1) -> None:
        """
        Sample new weights for the exploration matrix.

        :param n_envs:
        """
        assert isinstance(
            self.action_dist, StateDependentNoiseDistribution
        ), "reset_noise() is only available when using gSDE"
        self.action_dist.sample_weights(self.log_std, batch_size=n_envs)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        if isinstance(lr_schedule, tuple):
            actor_schedule, critic_schedule = lr_schedule
        else:
            actor_schedule, critic_schedule = lr_schedule, lr_schedule

        self.actor = nn.Sequential(
            nn.Linear(get_flattened_obs_dim(self.observation_space), 64),
            nn.Tanh(),
            nn.Linear(64, self.action_dist.action_dim),
        )
        self.actor.optimizer = self.optimizer_class(
            self.actor.parameters(), lr=actor_schedule(1), **self.optimizer_kwargs
        )

        self.critic = nn.Sequential(
            nn.Linear(
                get_flattened_obs_dim(self.observation_space)
                + self.action_dist.action_dim,
                64,
            ),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.critic.optimizer = self.optimizer_class(
            self.critic.parameters(), lr=critic_schedule(1), **self.optimizer_kwargs
        )

        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # Values from stable-baselines.
            module_gains = {
                self.actor: 0.01,
                self.critic: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

    def forward(
        self, obs: th.Tensor, deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        obs = obs.type(th.float32)
        values = self.critic(obs)
        mu = self.actor(obs)
        distribution = self._get_action_dist_from_latent(mu)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1,) + self.action_space.shape)
        return actions, values, log_prob

    def _get_action_dist_from_latent(self, mu: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param mu: mean of the distribution
        :return: Action distribution
        """
        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mu, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mu are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mu)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mu are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mu)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mu are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mu)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            raise NotImplementedError()
        else:
            raise ValueError("Invalid action distribution")

    def _predict(
        self, observation: th.Tensor, prev_noise: th.Tensor, episode_start:th.Tensor, deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param prev_noise: Noise of the previous action.
        :param episode_start: Bool tensor marking whether the obervation is first in the episode.
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy along with the used noise.
        """
        action_shape = observation.shape
        action_shape[-1] = self.action_dist.action_dim
        noise = self.noise_dist.sample(action_shape)

        #TODO: Check if applying the calculation on the whole tensor and only later overwriting start values doesn't speed up the algo.
        noise[~episode_start] = self.alpha * prev_noise[~episode_start] + self.alpha_sqrt * noise[~episode_start]

        observation = observation.type(th.float32)
        actions = self.get_distribution(observation).get_actions(
            deterministic=deterministic
        )
        return actions + noise

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        self.set_training_mode(False)

        observation, vectorized_env = self.obs_to_tensor(observation)

        with th.no_grad():
            actions, state = self._predict(
                observation, prev_noise=state, episode_start=episode_start, deterministic=deterministic
            )
        # Convert to numpy, and reshape to the original action shape
        actions = actions.cpu().numpy().reshape((-1,) + self.action_space.shape)

        if isinstance(self.action_space, gym.spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(
                    actions, self.action_space.low, self.action_space.high
                )

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions.squeeze(axis=0)

        return actions, state

    def evaluate_actions(
        self, obs: th.Tensor, actions: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        obs = obs.type(th.float32)
        mu = self.actor(obs)
        distribution = self._get_action_dist_from_latent(mu)
        log_prob = distribution.log_prob(actions)
        values = self.critic(obs)
        return values, log_prob, distribution.entropy()

    def get_distribution(self, obs: th.Tensor) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        obs = obs.type(th.float32)
        mu = self.actor(obs)
        return self._get_action_dist_from_latent(mu)

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs:
        :return: the estimated values.
        """
        obs = obs.type(th.float32)
        return self.critic(obs)
