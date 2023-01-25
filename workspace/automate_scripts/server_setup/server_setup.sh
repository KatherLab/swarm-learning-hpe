#!/bin/sh
set -eux

sudo apt install python3-pip -y
sudo pip install gdown

DIR="/apls-9.12/"
if [ -d "$DIR" ]; then
  # Take action if $DIR exists. #
  echo "APLS already installed"
else
  gdown --folder https://drive.google.com/drive/folders/166t3FPPtI25CQdNXkYtfX7thmXJcnOuO?usp=share_link  # for folders

fi

cd ./apls-9.12/UNIX
chmod a+x setup.bin
sudo ./setup.bin