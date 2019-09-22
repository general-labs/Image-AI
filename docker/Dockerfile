FROM ubuntu:16.04


MAINTAINER DIDARUL "didarul.amin@nbcuni.com"

RUN apt -y update &&\
    apt -y install python3 python3-pip &&\
    apt -y install libgtk2.0-dev 


RUN python3 -m pip install --upgrade pip

# ONLY FOR FACE_RECOGNITION
RUN apt-get install -y --fix-missing \
  build-essential \
  cmake \
  gfortran \
  wget \
  curl \
  graphicsmagick \
  libgraphicsmagick1-dev \
  libatlas-dev \
  libavcodec-dev \
  libavformat-dev \
  libgtk2.0-dev \
  libjpeg-dev \
  liblapack-dev \
  libswscale-dev \
  pkg-config \
  software-properties-common \
  zip

# Only For Smart Image Crop
RUN pip3 install https://github.com/OlafenwaMoses/ImageAI/releases/download/2.0.2/imageai-2.0.2-py3-none-any.whl

ADD ./python_requirements.txt /
RUN python3 -m pip install -r python_requirements.txt

# ADD ../scripts /
ENTRYPOINT ["bash","/scripts/start_flask.sh"]


