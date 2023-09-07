FROM python:3.8.12-slim-bullseye

WORKDIR /app

# Installing GCC
RUN apt update
RUN apt install -y build-essential

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN python -m spacy download en_core_web_sm
RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader punkt

# Copy local files to the container
COPY ./src/octis ./src/octis

# Setup octis
# It seems it has relative imports, so we need to change the working directory
WORKDIR /app/src/octis
RUN python setup.py install

# Get back to the root directory and copy the rest of the files
WORKDIR /app
COPY . .

# Changing working directory to src for later use at commandline
WORKDIR /app/src

RUN python ./main_exp_slim.py