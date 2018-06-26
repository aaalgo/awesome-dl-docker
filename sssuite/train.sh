#!/bin/bash

./run-docker.sh python main.py --mode train --model FC-DenseNet56 --dataset GMC --num_epochs 2 --crop_height 512 --crop_width 512
