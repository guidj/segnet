#!/usr/bin/env bash
wget https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz
mkdir -p input
mv vgg16_weights.npz input/

#pip install tqdm

mkdir -p ckpts
mkdir -p logs
# place files under input/raw dir
# use tfrecorder.py to generate images
# place files under input/{model-name}/... 
