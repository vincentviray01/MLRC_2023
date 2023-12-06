# MLRC_2023
MLRC 2023

# Steps
1. Download Docker Desktop (For Linux - https://docs.docker.com/desktop/install/linux-install/)
2. Install nvidia-container-toolkit (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

# Building Docker Container
$ docker build . -t mlrc_2023 -f docker/Dockerfile

# Opening Jupyter Notebooks (in Juypter Lab)
$ docker run --gpus all -it -p 8888:8888 -v `pwd`/data:/home/data -v `pwd`/src:/home/src mlrc_2023

And then click the link that looks like
http://127.0.0.1:8888/lab?token=80b9b0c1712265ed2049eb731caf0b1ea51c9074be06f599

# Running an experiment from commandline
$ papermill run_experiment.ipynb output.ipynb -p alpha 0.6 -p l1_ratio 0.1
