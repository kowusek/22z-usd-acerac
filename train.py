import gym
import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

from algos.acer import ACER

if __name__ == "__main__":
    env_id = 'HalfCheetah-v3'
    num_cpu = 6  # Number of processes to use
    # Create the vectorized environment
    env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)

    # currently a fake ACER (PPO in reality)
    model = ACER("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100_000)

    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
    