#!/bin/bash

CMD=$*

if [ -z "$CMD" ];
then
    CMD=/bin/bash
fi

TRAINING_DATA=$PWD/test/dataset
CHECKPOINT=$PWD/test/checkpoints
INFERENCE_INPUT=$PWD/test/input
INFERENCE_OUTPUT=$PWD/test/output

nvidia-docker run --rm -it -v $TRAINING_DATA:/sssuite/CamVid -v $INFERENCE_INPUT:/input -v $CHECKPOINT:/sssuite/checkpoints -v $INFERENCE_OUTPUT:/sssuite/Test aaalgo/sssuite $CMD

#nvidia-docker run --network=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $HOME/.Xauthority:/root/.Xauthority -v /shared/s2/users/lcai/Mask_RCNN_test/mask_rcnn_coco.h5:/Mask_RCNN/mask_rcnn_coco.h5 -v /shared/s2/users/lcai/awesome-dl-docker/mask-rcnn/test_fig:/test_fig --rm -it aaalgo/mask-rcnn  bash

