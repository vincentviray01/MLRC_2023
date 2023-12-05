from ubuntu:22.04


ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
	
WORKDIR /home

COPY requirements.yml requirements.yml
#RUN conda init && \
#	conda install --file requirements.yml

RUN conda env create -f requirements.yml

#COPY loss_functions.py loss_functions.py
#COPY utils.py utils.py
#COPY datasets.py datasets.py
#COPY run_evaluation.ipynb run_evaluation.ipynb
#COPY run_experiment.ipynb run_experiment.ipynb
#COPY run_training.ipynb run_training.ipynb

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "mlrc_2023", "jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]
# Host machine: docker run -it -p 8888:8888 image.version
