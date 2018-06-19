#Mask-RCNN docker 
Mask-RCNN implement by Matterport
https://github.com/matterport/Mask_RCNN

##Run a test
git clone this repository  

```
cd mask-rcnn
mkdir test_fig
chmod 777 test_fig

```
modify the path in run-docker.sh to mount your local files to docker container 

```
./run-docker.sh
```
It will make prediction on the sample images from pre-trained coco model 
