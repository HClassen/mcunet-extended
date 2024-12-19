FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu24.04

WORKDIR /mcunet

COPY ./src /mcunet/src
COPY ./main.py /mcunet/main.py
COPY ./pyproject.toml /mcunet/pyproject.toml

RUN apt update && apt install -y python3 python3-pip vim

RUN pip3 install -e . --break-system-packages
