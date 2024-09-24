#!/bin/bash

# Allow local connections to the X server for GUI applications in Docker
xhost +local:docker

# Setup for X11 forwarding to enable GUI
XAUTH=/tmp/.docker.xauth
touch $XAUTH
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

IMAGE_NAME=colorcloud_ufgsim:latest

# Define the container's folder path where the host folder will be mounted (ONLY FOR DEVELOPMENT)
HOST_FOLDER_PATH="$(pwd)"
CONTAINER_FOLDER_PATH="/root/ros2_ws/src/ufgsim_pkg"

# Run the Docker container with the selected image and configurations for GUI applications
docker run -it \
  --user=root \
  --privileged \
  --network=host \
  --ipc=host \
  --pid=host \
  --env="DISPLAY=$DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --env="XAUTHORITY=$XAUTH" \
  --volume="$XAUTH:$XAUTH" \
  --runtime nvidia \
  --gpus all \
  --env="NVIDIA_VISIBLE_DEVICES=all" \
  --env="NVIDIA_DRIVER_CAPABILITIES=all" \
  --volume="$HOST_FOLDER_PATH:$CONTAINER_FOLDER_PATH:rw"
  $IMAGE_NAME