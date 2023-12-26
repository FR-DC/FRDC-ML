FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime as torch
WORKDIR /devcontainer

COPY ./pyproject.toml /devcontainer/pyproject.toml

RUN apt-get update
RUN apt-get install git -y

RUN pip3 install --upgrade pip && \
    pip3 install poetry

RUN conda init bash \
    && . ~/.bashrc \
    && conda activate base \
    && poetry config virtualenvs.create false \
    && poetry install --with dev --no-interaction --no-ansi

RUN apt-get install curl -y && curl -sSL https://sdk.cloud.google.com | bash
ENV PATH $PATH:/root/google-cloud-sdk/bin