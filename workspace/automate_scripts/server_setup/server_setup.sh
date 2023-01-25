#!/bin/sh
set -eux

sudo apt install python3-pip
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1Y6DR-7Gp1uWsqQ1CDZGb6cxMD34-NOtG  # for folders

cd ./pls-9.12/UNIX
chmod a+x setup.bin
sudo ./setup.bin