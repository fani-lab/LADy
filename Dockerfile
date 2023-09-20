FROM python:3.8.12-slim-bullseye

WORKDIR /app

# Installing GCC
RUN apt update

RUN apt install -y build-essential

# Install dependencies
RUN pip install pipx

# RUN pipx ensurepath
ENV PATH=/root/.local/bin:$PATH

RUN pipx install poetry==1.6.0
RUN pipx install poethepoet

COPY pyproject.toml .

RUN poetry install

COPY . .

RUN poe post_install

# Changing working directory to src for later use at commandline
WORKDIR /app/src

RUN poe dummy

CMD [ "/bin/bash" ]
