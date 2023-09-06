#!/bin/sh
directory=./data/RealSR
#file="./data/RealSR/RealSR(V3).tar.gz"
file="./data/RealSR/RealSR(V3).tar.gz"
#file="./data/RealSR/RealSR(Final).tar.gz"
mkdir -p $directory
#gdown -O $file --id 17ZMjo-zwFouxnm_aFM6CUHBwgRrLZqIM
gdown -O $file --id 17ZMjo-zwFouxnm_aFM6CUHBwgRrLZqIM
#gdown -O $file --id 1dEBRo_1HH6Yk9zrchEg_JTRi-Uhmd-sj
tar -xzf $file -C $directory && rm $file
