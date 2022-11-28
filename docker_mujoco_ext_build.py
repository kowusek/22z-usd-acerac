from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

# Force build mujoco in docker build by importing the HalfCheetah-v3 environment.

if __name__ == "__main__":
    env_id = 'HalfCheetah-v3'
    env = make_vec_env(env_id, n_envs=1, seed=0, vec_env_cls=SubprocVecEnv)