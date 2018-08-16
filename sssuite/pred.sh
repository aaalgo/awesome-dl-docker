#!/bin/bash  
for file in ../input/*.png; do python ./main.py --mode predict --model FC-DenseNet56 --image $file; done 
