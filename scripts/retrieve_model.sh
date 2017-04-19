#!/bin/bash

echo "Downloading model weights"
wget -O ./models/ResNet50.npy https://lexiondebug.blob.core.windows.net/mlmodel/models/res50.npy
wget -O ./models/VGG16.npy https://lexiondebug.blob.core.windows.net/mlmodel/models/VGG_16.npy
wget -O ./models/AlexNet.npy http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy
echo "Please goto https://jumpshare.com/v/2OSrR2ePBf5z9VyFBx2T to download CaffeNet.npy to ./models/CaffeNet.npy"
