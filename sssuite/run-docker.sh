#!/bin/bash

CMD=$*

if [ -z "$CMD" ];
then
    CMD=/bin/bash
fi

nvidia-docker run --rm -it -v $PWD/test/dataset:/sssuite/CamVid -v $PWD/test/input:/input -v $PWD/test/checkpoints:/sssuite/checkpoints -v $PWD/test/output:/sssuite/Test aaalgo/sssuite $CMD

#nvidia-docker run --network=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $HOME/.Xauthority:/root/.Xauthority -v /shared/s2/users/lcai/Mask_RCNN_test/mask_rcnn_coco.h5:/Mask_RCNN/mask_rcnn_coco.h5 -v /shared/s2/users/lcai/awesome-dl-docker/mask-rcnn/test_fig:/test_fig --rm -it aaalgo/mask-rcnn  bash

