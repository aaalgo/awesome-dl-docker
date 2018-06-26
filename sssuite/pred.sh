#!/bin/bash 
for file in ./Pred/*.png; do python ./main.py --mode predict --dataset GMC --model PSPNet-Res50 --image $file; done 
