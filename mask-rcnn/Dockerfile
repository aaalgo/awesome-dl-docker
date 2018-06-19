from tensorflow/tensorflow:1.6.0-gpu-py3 
ADD requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt
RUN apt-get update && apt-get install -y vim cmake protobuf-compiler libgtk2.0 python3-tk git cmake python3-setuptools eog
RUN cd / && rm -rf /Mask_RCNN && git clone --depth 2 https://github.com/matterport/Mask_RCNN.git && cd /Mask_RCNN && git checkout cbff80f3e3f653a9eeee43d0d383a0385aba546b && python3 setup.py install 
RUN cd / && rm -rf /coco && git clone https://github.com/pdollar/coco.git && cd /coco && git checkout ed842bffd41f6ff38707c4f0968d2cfd91088688 && cd /coco/PythonAPI/ && python3 setup.py build_ext  
WORKDIR /Mask_RCNN
ADD pred_demo.py /Mask_RCNN 

