#!/bin/sh
set -eux

cd "/opt/HP/HP AutoPass License Server/HP AutoPass License Server/HP AutoPass License Server/conf"

sudo sed -i "s+5814+5000+g" server.xml

cd "/opt/HP/HP AutoPass License Server/HP AutoPass License Server/HP AutoPass License Server/bin"
sudo cp hpLicenseServer  /etc/init.d/hpLicenseServer
sudo chmod 755 /etc/init.d/hpLicenseServer
cd /etc/init.d
sudo update-rc.d hpLicenseServer defaults 97 03
service hpLicenseServer start
service hpLicenseServer status