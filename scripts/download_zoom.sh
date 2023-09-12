#!/bin/sh
directory=./data/Zoom
#file="./data/RealSR/RealSR(V3).tar.gz"
file="./data/Zoom/train.zip"
#file="./data/RealSR/RealSR(Final).tar.gz"
mkdir -p $directory
#gdown -O $file --id 17ZMjo-zwFouxnm_aFM6CUHBwgRrLZqIM
gdown -O $file --id 1qp6z3F4Ru9srwq1lNZr3pQ4kcVN-AOlM
# url='https://drive.google.com/file/d/1qp6z3F4Ru9srwq1lNZr3pQ4kcVN-AOlM/view?usp=drive_link'
# gdown $url
#gdown -O $file --id 1dEBRo_1HH6Yk9zrchEg_JTRi-Uhmd-sj
tar -xzf $file -C $directory && rm $file


# https://drive.google.com/file/d/1qp6z3F4Ru9srwq1lNZr3pQ4kcVN-AOlM/view?usp=drive_link

gdown https://drive.google.com/uc?id=1qp6z3F4Ru9srwq1lNZr3pQ4kcVN-AOlM

gdown "<1qp6z3F4Ru9srwq1lNZr3pQ4kcVN-AOlM>&confirm=t"