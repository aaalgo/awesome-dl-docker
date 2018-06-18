#!/bin/bash

./run-docker.sh python3 avod/experiments/run_inference.py --checkpoint_name=pyramid_cars_with_aug_example --data_split=test --ckpt_indices=120 --device=0
./run-docker.sh python3 avod/data/save_kitti_predictions.py
