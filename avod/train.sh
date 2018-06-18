#!/bin/bash

./run-docker.sh python3 scripts/preprocessing/gen_mini_batches.py
./run-docker.sh python3 avod/experiments/run_training.py --pipeline_config=avod/configs/pyramid_cars_with_aug_example.config
