docker build . -t mlrc_2023 -f docker/Dockerfile;docker run --gpus all -it -p 8888:8888 -v `pwd`/data:/home/data -v `pwd`/src:/home/src mlrc_2023
