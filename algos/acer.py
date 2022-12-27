import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import gym
import numpy as np
import torch as th
from torch.nn import functional as F

from copy import deepcopy
from algos.pi_buffer import PiReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.policies import BasePolicy, ActorCriticPolicy
from stable_baselines3.common.type_aliases import (
    GymEnv,
    MaybeCallback,
    RolloutReturn,
    Schedule,
    TrainFreq,
    TrainFrequencyUnit,
)
from stable_baselines3.common.utils import (
    get_linear_fn,
    get_parameters_by_name,
    should_collect_more_steps,
)
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.dqn.policies import (
    CnnPolicy,
    MlpPolicy,
    MultiInputPolicy,
)


ACERSelf = TypeVar("ACERSelf", bound="ACER")


class ACER(OffPolicyAlgorithm):
    """
    The ACER (Actor-Critic with Experience Replay) model class, https://arxiv.org/abs/1611.01224

    !TODO: code copied from DQM. The params prepended with `--` are not yet tested and may not be needed

    --:param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    --:param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    --:param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    --:param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
    --:param gamma: the discount factor
    --:param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    --:param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    --:param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    --:param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    --:param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    --:param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
    --:param exploration_initial_eps: initial value of random action probability
    --:param exploration_final_eps: final value of random action probability
    --:param max_grad_norm: The maximum value for the gradient clipping
    --:param tensorboard_log: the log location for tensorboard (if None, no logging)
    --:param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically (Only available when passing string for the environment).
        Caution, this parameter is deprecated and will be removed in the future.
        Please use `EvalCallback` or a custom Callback instead.
    --:param policy_kwargs: additional arguments to be passed to the policy on creation
    --:param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    --:param seed: Seed for the pseudo random generators
    --:param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    --:param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": ActorCriticPolicy,
    }

    policy: ActorCriticPolicy
    replay_buffer: PiReplayBuffer

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 2e-5,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 5000,
        batch_size: int = 32,
        tau: float = 0.05,  # min{policy_frac, b}; b = tau
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[Type[PiReplayBuffer]] = PiReplayBuffer,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=None,  # No action noise
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(gym.spaces.Box,),
            support_multi_env=True,
        )

        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        # For updating the target network with multiple envs:
        self._n_calls = 0
        self.max_grad_norm = max_grad_norm
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = 0.0
        # Linear schedule will be defined in `_setup_model()`
        self.exploration_schedule = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )

    def _create_aliases(self) -> None:
        pass
        # self.actor = self.policy.actor
        # self.critic = self.policy.critic

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:

        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        v_losses = []
        p_losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )

            current_distributions = self.policy.get_distribution(
                replay_data.observations
            )
            current_action_log_probs = current_distributions.log_prob(
                replay_data.actions
            )

            # Get current state value (critic) estimates
            current_values = self.policy.predict_values(replay_data.observations)

            with th.no_grad():
                next_values = self.policy.predict_values(replay_data.next_observations)
                # Avoid potential broadcast issue
                next_values = next_values.reshape(-1, 1)

                target_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * self.gamma * next_values
                )
                advantage = target_values - current_values

                policy_proximity_coeff: th.Tensor = th.minimum(
                    th.exp(
                        current_action_log_probs - replay_data.log_probs
                    ),  # ? detach replay_data.log_probs?
                    th.full(
                        size=current_action_log_probs.shape,
                        fill_value=self.tau,
                        device=self.device,
                    ),
                ).detach()

            policy_loss = (
                advantage.detach()  # ? detach() needed?
                * current_action_log_probs  # requires_grad=True
                * policy_proximity_coeff  # detached scalar
            ).mean()

            v_loss = (
                advantage.detach()  # ? detach() needed?; detached scalar
                * current_values  # requires_grad=True
                * policy_proximity_coeff  # detached scalar
            ).mean()

            loss = policy_loss + v_loss
            losses.append(loss.item())
            v_losses.append(v_loss.item())
            p_losses.append(policy_loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
        self.logger.record("train/v_loss", np.mean(v_losses))
        self.logger.record("train/p_loss", np.mean(p_losses))

    def learn(
        self: ACERSelf,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "ACER",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> ACERSelf:

        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    ############ Minor framework overrides ############

    # overwrites OffPolicyAlgorithm to store log_probs of actions
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: PiReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``PiReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert (
                train_freq.unit == TrainFrequencyUnit.STEP
            ), "You must use only one env when doing episodic training."

        # Vectorize action noise if needed
        if (
            action_noise is not None
            and env.num_envs > 1
            and not isinstance(action_noise, VectorizedActionNoise)
        ):
            action_noise = VectorizedActionNoise(action_noise, env.num_envs)

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True

        while should_collect_more_steps(
            train_freq, num_collected_steps, num_collected_episodes
        ):
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and num_collected_steps % self.sde_sample_freq == 0
            ):
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(
                learning_starts, action_noise, env.num_envs
            )

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(
                    num_collected_steps * env.num_envs,
                    num_collected_episodes,
                    continue_training=False,
                )

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            with th.no_grad():
                obs_t = th.Tensor(self._last_obs, device=self.device)
                actions_t = th.Tensor(actions, device=self.device)
                dist = self.policy.get_distribution(obs_t)
                log_probs = dist.log_prob(
                    actions_t
                )  # ? check if sum_independent_dims() should really be called
            self._store_transition(
                replay_buffer, buffer_actions, log_probs, new_obs, rewards, dones, infos
            )

            self._update_current_progress_remaining(
                self.num_timesteps, self._total_timesteps
            )

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if (
                        log_interval is not None
                        and self._episode_num % log_interval == 0
                    ):
                        self._dump_logs()

        callback.on_rollout_end()

        return RolloutReturn(
            num_collected_steps * env.num_envs,
            num_collected_episodes,
            continue_training,
        )

    # overwrites OffPolicyAlgorithm to store log_probs of actions
    def _store_transition(
        self,
        replay_buffer: PiReplayBuffer,
        buffer_action: np.ndarray,
        log_probs: np.ndarray,
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """
        Store transition in the replay buffer.
        We store the normalized action and the unnormalized observation.
        It also handles terminal observations (because VecEnv resets automatically).

        :param replay_buffer: Replay buffer object where to store the transition.
        :param buffer_action: normalized action
        :param new_obs: next observation in the current episode
            or first observation of the episode (when dones is True)
        :param reward: reward for the current transition
        :param dones: Termination signal
        :param infos: List of additional information about the transition.
            It may contain the terminal observations and information about timeout.
        """
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # Avoid modification by reference
        next_obs = deepcopy(new_obs_)
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    # Replace next obs for the correct envs
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(
                            next_obs[i, :]
                        )

        replay_buffer.add(
            self._last_original_obs,
            next_obs,
            buffer_action,
            log_probs,
            reward_,
            dones,
            infos,
        )

        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_
