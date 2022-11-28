Dockerfile provided for setting up the environment. All scripts are recommended to be run from inside the container.

```sh
xhost +local:root
docker build -t usd -f Dockerfile .
docker run -it --rm -v "$(pwd)":/p -v /tmp/.X11-unix:/tmp/.X11-unix:ro --network host --ipc=host usd
```

Example `train.py` run (from inside the docker!):
```sh
python train.py
```
