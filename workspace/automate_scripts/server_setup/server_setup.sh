#!/bin/sh
set -eux

sudo apt install python3-pip -y
sudo pip install gdown
gdown --folder https://drive.google.com/drive/folders/166t3FPPtI25CQdNXkYtfX7thmXJcnOuO?usp=share_link  # for folders

cd ./apls-9.12/UNIX
chmod a+x setup.bin
sudo ./setup.bin