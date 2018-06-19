from tensorflow/tensorflow:1.4.1-gpu-py3
RUN pip3 install opencv-python shapely numba easydict cython
RUN apt-get update && apt-get install -y git vim
#RUN cd / && rm -rf voxelnet \
#    && git clone https://github.com/qianguih/voxelnet \
#    && cd /voxelnet \
#    && git checkout b74823daa328fc2fa99452bf79793e1f3c32c72a \
#    && python3 setup.py build_ext --inplace \
#    && cd kitti_eval \
#    && g++ -o evaluate_object_3d_offline evaluate_object_3d_offline.cpp


WORKDIR /voxelnet
   
