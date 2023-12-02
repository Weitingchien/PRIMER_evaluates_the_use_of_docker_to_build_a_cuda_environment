FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

RUN apt-get update && apt-get install -y wget git && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda && \
    rm -rf /var/lib/apt/lists/* \
    rm -f Miniconda3-latest-Linux-x86_64.sh \
    rm -rf /root/.cache/huggingface/datasets/downloads/extracted/ \
    rm -rf /root/.cache/huggingface/transformers \
    rm -rf /root/.cache/huggingface/datasets

ENV PATH="/miniconda/bin:${PATH}"

WORKDIR /app

COPY . /app

RUN conda env create -f environment.yml

CMD ["conda", "run", "-n", "primer", "python", "main.py"]


