#!/bin/bash

CMD=$*

if [ -z "$CMD" ];
then
    CMD=/bin/bash
fi

KITTI_DIR=      #kitti directory, containing training and testing
CROPPED_DIR=    #cropped data

if [ ! -d $KITTI_DIR/training ]
then
    echo Please edit and set KITTI_DIR
    exit
fi

if [ ! -d $CROPPED_DIR ]
then
    echo Please edit and set CROPPED_DIR
    exit
fi

nvidia-docker run  \
        -v $KITTI_DIR:/voxelnet/data/object \
        -v $KITTI_DIR/training/calib:/media/hdc/KITTI/calib/data_object_calib/training/calib \
        -v $KITTI_DIR/training/image_2:/media/hdc/KITTI/image/training/image_2/ \
        -v $KITTI_DIR/training/velodyne:/media/hdc/KITTI/point_cloud/raw_bin_files/training/velodyne/ \
        -v $CROPPED_DIR:/media/hdc/KITTI/for_voxelnet/cropped_dataset \
        --rm -it aaalgo/voxelnet $CMD

