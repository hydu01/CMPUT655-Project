FROM nvidia/cuda:11.7.1-runtime-ubuntu20.04
LABEL maintainer="haruto@ualberta.ca"

ENV DEBIAN_FRONTEND=noninteractive
ENV WORKDIR=/workspace
RUN mkdir -p $WORKDIR && \
	chmod 777 $WORKDIR

WORKDIR $WORKDIR

RUN apt-get update && \
	apt-get -y install curl python3-pip git build-essential python-opengl software-properties-common && \
	add-apt-repository -y ppa:deadsnakes/ppa && \
	apt-get install -y python3.10
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip install poetry --upgrade
COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock
RUN poetry install

# COPY entrypoint.sh /usr/local/bin
# RUN chmod 777 /usr/local/bin/entrypoint.sh
# ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
ENTRYPOINT ["/bin/bash"]

COPY ./codes $WORKDIR/codes
