#!/bin/bash
nvidia-docker run --network=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $HOME/.Xauthority:/root/.Xauthority -v /shared/s2/users/lcai/Mask_RCNN_test/mask_rcnn_coco.h5:/Mask_RCNN/mask_rcnn_coco.h5 -v /shared/s2/users/lcai/awesome-dl-docker/mask-rcnn/test_fig:/test_fig --rm -it aaalgo/mask-rcnn  bash
