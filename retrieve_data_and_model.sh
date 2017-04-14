#!/bin/bash

sudo apt-get install -y axel

mkdir ./data/test_data
echo "Downloading ILSVRC2012 dataset"
axel -a -n 16 -o ./data/test_data http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar
tar -xvzf ./data/test_data/ILSVRC2012_img_val.tar

echo "Downloading model weights"
axel -a -n 16 -o ./models/ResNet50.npy https://lexiondebug.blob.core.windows.net/mlmodel/models/res50.npy
axel -a -n 16 -o ./models/VGG16.npy https://lexiondebug.blob.core.windows.net/mlmodel/models/VGG_16.npy
axel -a -n 16 -o ./models/AlexNet.npy http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy
echo "Please goto https://jumpshare.com/v/2OSrR2ePBf5z9VyFBx2T to download CaffeNet.npy to ./models/CaffeNet.npy"
