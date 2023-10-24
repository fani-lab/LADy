FROM python:3.8.12-slim-bullseye

WORKDIR /app

# Installing GCC
RUN apt update

RUN apt install -y build-essential

# Installing Node.js
ENV NODE_VERSION=20.9.0
RUN apt install -y curl
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
ENV NVM_DIR=/root/.nvm
RUN . "$NVM_DIR/nvm.sh" && nvm install ${NODE_VERSION}
RUN . "$NVM_DIR/nvm.sh" && nvm use v${NODE_VERSION}
RUN . "$NVM_DIR/nvm.sh" && nvm alias default v${NODE_VERSION}
ENV PATH="/root/.nvm/versions/node/v${NODE_VERSION}/bin/:${PATH}"
RUN node --version
RUN npm --version

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
