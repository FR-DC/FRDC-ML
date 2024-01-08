FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime as torch
WORKDIR /devcontainer

COPY ./pyproject.toml /devcontainer/pyproject.toml

RUN apt update -y && apt upgrade -y
RUN apt install git -y

RUN pip3 install --upgrade pip && \
    pip3 install poetry && \
    pip3 install lightning

RUN conda init bash \
    && . ~/.bashrc \
    && conda activate base \
    && poetry config virtualenvs.create false \
    && poetry install --with dev --no-interaction --no-ansi

RUN apt install curl -y && curl -sSL https://sdk.cloud.google.com | bash
ENV PATH $PATH:/root/google-cloud-sdk/bin
