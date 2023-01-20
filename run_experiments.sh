python3.7 train.py --seed=123 --model=ACER --num_cpu=1 --timesteps=10_000_000 --record_video --save_path="ACER_123/model"
python3.7 train.py --seed=234 --model=ACER --num_cpu=1 --timesteps=10_000_000 --record_video --save_path="ACER_234/model"
python3.7 train.py --seed=345 --model=ACER --num_cpu=1 --timesteps=10_000_000 --record_video --save_path="ACER_345/model"

python3.7 train.py --seed=123 --model=PPO --num_cpu=6 --timesteps=4_000_000 --record_video --save_path="PPO_4kk_123/model"
python3.7 train.py --seed=234 --model=PPO --num_cpu=6 --timesteps=4_000_000 --record_video --save_path="PPO_4kk_234/model"
python3.7 train.py --seed=345 --model=PPO --num_cpu=6 --timesteps=4_000_000 --record_video --save_path="PPO_4kk_345/model"

python3.7 train.py --seed=123 --model=SAC --num_cpu=6 --timesteps=4_000_000 --record_video --save_path="SAC_4kk_123/model"
python3.7 train.py --seed=234 --model=SAC --num_cpu=6 --timesteps=4_000_000 --record_video --save_path="SAC_4kk_234/model"
python3.7 train.py --seed=345 --model=SAC --num_cpu=6 --timesteps=4_000_000 --record_video --save_path="SAC_4kk_345/model"