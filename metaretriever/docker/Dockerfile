FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04
LABEL maintainer="Yaojie Lu"
LABEL repository="uie"

RUN apt update && \
    apt install -y bash \
                   build-essential \
                   git \
                   curl \
                   ca-certificates \
                   python3 \
                   python3-pip && \
    rm -rf /var/lib/apt/lists

WORKDIR /pre_env

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir mkl && \
    python3 -m pip install --no-cache-dir torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html

RUN git clone https://github.com/NVIDIA/apex
RUN cd apex && \
    python3 setup.py install && \
    python3 -m pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

COPY ./requirements.txt .
RUN python3 -m pip install -r ./requirements.txt

CMD ["/bin/bash"]
