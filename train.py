import gym
import numpy as np
import torch as th
import random
import argparse

from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecVideoRecorder,
)
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.sac.sac import SAC
import wandb
from wandb.integration.sb3 import WandbCallback

from algos.acer import ACER
from algos.acerac import ACERAC

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="USD ACER Half-Cheetah training script",
    )
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-t", "--timesteps", type=int, default=1_000_000)
    parser.add_argument(
        "-m", "--model", type=str, default="ACER", choices=["ACER", "PPO", "SAC"]
    )
    parser.add_argument(
        "--num_cpu", type=int, default=1, help="number of env processes to use"
    )
    parser.add_argument("-w", "--wandb", type=bool, default=True)
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--load_path", type=str)
    parser.add_argument("--save_path", type=str)
    args = parser.parse_args()

    if args.wandb:
        wandb_run = wandb.init(
            project="usd_acer",
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=False,
            name=f"{args.model}_{args.seed}",
            config=args,
        )
        callback = WandbCallback(gradient_save_freq=2)
    else:
        callback = None

    env_id = "HalfCheetah-v3"
    num_cpu = args.num_cpu  # Number of processes to use
    # Create the vectorized environment

    seed = args.seed

    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)

    env = make_vec_env(
        env_id, n_envs=args.num_cpu, seed=seed, vec_env_cls=SubprocVecEnv
    )
    # env = VecCheckNan(env, raise_exception=True)

    model_cls = ACER
    if args.model == "PPO":
        model_cls = PPO
    if args.model == "SAC":
        model_cls = SAC

    if args.load_path is None:
        model = model_cls(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=f"runs/{wandb_run.id}",
            seed=seed,
        )
    else:
        model = model_cls.load(args.load_path)

    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.use_deterministic_algorithms(True)

    model.learn(total_timesteps=args.timesteps, callback=callback)

    if args.save_path is not None:
        model.save(args.save_path)

    if args.wandb:
        wandb_run.finish()

    if args.record_video:
        print("recording video...")
        video_length = 1000
        obs = env.reset()
        # Record the video starting at the first step
        env = VecVideoRecorder(
            env,
            f"{args.model}_{seed}.mp4",
            record_video_trigger=lambda x: x == 0,
            video_length=video_length,
            name_prefix=f"{args.model}_{seed}_env{env_id}",
        )
        for _ in range(video_length + 1):
            action = model.predict(obs, deterministic=True)
            obs, _, _, _ = env.step(action)
        # Save the video
        env.close()
