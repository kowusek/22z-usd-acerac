FROM stablebaselines/rl-baselines3-zoo-cpu:1.5.1a6

RUN  apt-get update \
  && apt-get install -y wget
RUN apt install libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf -y
# fixes warnings related to urllib3
RUN python -m pip install requests==2.28.1

ENV MUJOCO_PY_MUJOCO_PATH /mujoco/mujoco210
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/mujoco/mujoco210/bin"
RUN wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz && mkdir /mujoco && tar -xvzf ./mujoco210-linux-x86_64.tar.gz -C /mujoco
RUN python -m pip install mujoco-py==2.1.2.14
RUN python -m pip install stable-baselines3==1.6.2
RUN python -m pip install wandb==0.13.3

COPY docker_mujoco_ext_build.py /docker_mujoco_ext_build.py
RUN python /docker_mujoco_ext_build.py

WORKDIR /p