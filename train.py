import gym
import numpy as np
import torch as th
import random

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecCheckNan
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.sac.sac import SAC
import wandb
from wandb.integration.sb3 import WandbCallback

from algos.acer import ACER

if __name__ == "__main__":
    wandb_run = wandb.init(
        project="usd_acer",
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=False,
    )

    env_id = "HalfCheetah-v3"
    num_cpu = 1  # Number of processes to use
    # Create the vectorized environment

    seed = 0

    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)

    env = make_vec_env(env_id, n_envs=num_cpu, seed=seed, vec_env_cls=SubprocVecEnv)
    # env = VecCheckNan(env, raise_exception=True)

    model = ACER(
        "MlpPolicy", env, verbose=1, tensorboard_log=f"runs/{wandb_run.id}", seed=seed
    )

    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.use_deterministic_algorithms(True)

    model.learn(total_timesteps=1_000_000, callback=WandbCallback(gradient_save_freq=2))
    wandb_run.finish()

    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
