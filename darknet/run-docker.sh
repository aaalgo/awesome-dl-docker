#!/bin/bash

CMD=$*

if [ -z "$CMD" ];
then
    CMD=/bin/bash
fi

TRAINING_DATA=$PWD/dataset/trainval/train
VALIDATION_DATA=$PWD/dataset/trainval/val  
TRAINING_OUTPUT=$PWD/weights
PREDICTION=$PWD/results

nvidia-docker run --rm -it \
    -v $TRAINING_DATA:/darknet/trainval/train \
    -v $VALIDATION_DATA:/darknet/trainval/val \
    -v $TRAINING_OUTPUT:/darknet/weights \
    -v $PREDICTION:/darknet/results \
    aaalgo/darknet $CMD


