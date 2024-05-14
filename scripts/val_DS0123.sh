#!/bin/sh

# 執行以下命令
#/home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data 設為一個變數
path=/home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data
host=./trained_models/best.pth
server=./trained_models/server/best.pth

python val.py -ti $path/DS01/images/ -tm $path/DS01/masks/ -m $host
python val.py -ti $path/DS02/images/ -tm $path/DS02/masks/ -m $host
python val.py -ti $path/DS03/images/ -tm $path/DS03/masks/ -m $host
python val.py -ti $path/Real/images/ -tm $path/Real/masks/ -m $host

python val.py -ti $path/DS01/images/ -tm $path/DS01/masks/ -m $server
python val.py -ti $path/DS02/images/ -tm $path/DS02/masks/ -m $server
python val.py -ti $path/DS03/images/ -tm $path/DS03/masks/ -m $server
python val.py -ti $path/Real/images/ -tm $path/Real/masks/ -m $server


