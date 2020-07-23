FROM ubuntu:16.04

#set up environment
RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y python3.6 python3.6-dev python3-pip
RUN pip3 install --upgrade pip

WORKDIR /workspace/
RUN apt-get update && apt-get install -y git unzip zip wget nano default-jre
RUN wget http://nlp.stanford.edu/software/stanford-corenlp-full-2016-10-31.zip
RUN unzip stanford-corenlp-latest.zip
ENV CLASSPATH=/workspace/stanford-corenlp-full-2016-10-31/stanford-corenlp-3.7.0.jar

COPY cnn_stories.tgz .
COPY dailymail_stories.tgz .
RUN tar zxvf cnn_stories.tgz && tar zxvf dailymail_stories.tgz