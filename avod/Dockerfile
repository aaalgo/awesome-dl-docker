from tensorflow/tensorflow:1.3.0-gpu-py3
ADD requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt
RUN apt-get update && apt-get install -y vim cmake protobuf-compiler libgtk2.0 python3-tk git cmake
ENV PYTHONPATH /avod:/avod/wavedata
RUN cd / && rm -rf /avod \
    && git clone https://github.com/kujason/avod \
    && cd avod && git checkout b7dcf31e50230c67bcf19ed063549a8b1bc4a785 \
    && git clone https://www.github.com/kujason/wavedata \
    && cd wavedata && git checkout c4b5aabd9eb3b74fad777349f75161032d3460fa \
    && cd /avod && sh scripts/install/build_integral_image_lib.bash \
    && sh avod/protos/run_protoc.sh

