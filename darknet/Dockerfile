from nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04
RUN apt-get update && apt-get install -y git vim libboost-all-dev g++
RUN apt-get install -y libopencv-dev
# checkout from non-official repo for the compiling fix
# need to switch to official repo when the fix is merged
RUN cd / && rm -rf darknet \
    && git clone https://github.com/pjreddie/darknet \
    && cd /darknet \
    && git checkout f61f100b2ef4e963bc43deb0cf995364e2de40b4 \
    && make 

WORKDIR /darknet
   
