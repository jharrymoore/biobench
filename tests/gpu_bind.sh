#!/bin/bash


# create an array like 0,0,1,1,2,2,3,3 up to CUDA_VISIBLE_DEVICES
a=(0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7)
device=${a[$OMPI_COMM_WORLD_LOCAL_RANK]}

# index into the array
export CUDA_VISIBLE_DEVICES=$device
echo $device
# export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK



$@
