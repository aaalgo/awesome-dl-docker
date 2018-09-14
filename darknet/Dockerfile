from nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
RUN apt-get update && apt-get install -y git vim libboost-all-dev g++ wget python-pip
RUN apt-get install -y libopencv-dev libgtk2.0 python-tk
RUN pip install numpy matplotlib opencv-python
# checkout from non-official repo for the compiling fix
# need to switch to official repo when the fix is merged
RUN cd / && rm -rf darknet \
    && git clone https://github.com/pjreddie/darknet \
    && cd /darknet \
    && git checkout f61f100b2ef4e963bc43deb0cf995364e2de40b4 \
    && make 
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:/darknet/
WORKDIR /darknet
   
