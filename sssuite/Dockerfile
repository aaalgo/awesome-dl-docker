FROM tensorflow/tensorflow:1.6.0-gpu

# Install Python Dependencies
RUN pip install numpy matplotlib opencv-python scipy sklearn
RUN apt-get update && apt-get install -y vim git libgtk2.0 python-tk

# Clone the SSSuite repository
RUN git clone https://github.com/GeorgeSeif/Semantic-Segmentation-Suite.git /sssuite && cd /sssuite && git checkout e17c527c3e40d6bb611ca1e3ca7d637f3235d4d1
  
ADD pred.sh /sssuite/pred.sh
ADD main.py /sssuite/main.py
# Go to Detectron root
WORKDIR /sssuite
