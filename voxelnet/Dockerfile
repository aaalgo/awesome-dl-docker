from tensorflow/tensorflow:1.4.1-gpu-py3
RUN pip3 install opencv-python shapely numba easydict cython
RUN apt-get update && apt-get install -y git vim libboost-all-dev libsm6 libxext6 libxrender-dev
# checkout from non-official repo for the compiling fix
# need to switch to official repo when the fix is merged
RUN cd / && rm -rf voxelnet \
    && git clone https://github.com/bw4sz/voxelnet \
    && cd /voxelnet \
    && git checkout 9ddd143c22819052363103b8b3942d0ec6b586fd \
    && python3 setup.py build_ext --inplace \
    && cd kitti_eval \
    && g++ -o evaluate_object_3d_offline evaluate_object_3d_offline.cpp


WORKDIR /voxelnet
   
