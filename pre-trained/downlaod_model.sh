#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define variables
IMAGE_NAME="mmdnn/mmdnn:cpu.small"
CONTAINER_NAME="mmdnn-convert"
MODEL_DIR="hybrid"

# Pull the Docker image
docker pull $IMAGE_NAME

# Run the Docker container
docker run --name $CONTAINER_NAME $IMAGE_NAME /bin/bash -c "
set -x &&
mkdir $MODEL_DIR && 
cd $MODEL_DIR &&
wget http://places.csail.mit.edu/model/hybridCNN_upgraded.tar.gz &&
gunzip < hybridCNN_upgraded.tar.gz | tar xvf - &&
rm *.csv *.binaryproto *.tar.gz &&
mmconvert -sf caffe -in hybridCNN_deploy_upgraded.prototxt -iw hybridCNN_iter_700000_upgraded.caffemodel -df tensorflow -om output_model.pb
"

# Copy the model files to the script's directory (pre-trained folder)
docker cp $CONTAINER_NAME:/mmdnn/$MODEL_DIR $SCRIPT_DIR/

# Rename .npy file to weights.npy in the script's directory
mv $SCRIPT_DIR/$MODEL_DIR/*.npy $SCRIPT_DIR/$MODEL_DIR/weights.npy

# Clean up the container and image
docker stop $CONTAINER_NAME
docker rm $CONTAINER_NAME
docker rmi $IMAGE_NAMEe