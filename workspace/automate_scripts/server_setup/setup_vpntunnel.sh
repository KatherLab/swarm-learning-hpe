#!/bin/sh
set -eux

# Help function
help() {
   echo ""
   echo "Usage: setup_vpntunnel.sh"
   echo ""
   exit 1
}

# Process command options
while getopts "h?" opt
do
   case "$opt" in
      h ) help ;;
      ? ) help ;;
   esac
done

echo setup vpn tunnel for swarm learning
sudo apt-get -y install openvpn
if ! sudo ls /etc/openvpn/credentials >/dev/null 2>&1; then
    # The file exists, continue with the following commands
    sudo touch /etc/openvpn/credentials
fi
sudo chmod 777 /etc/openvpn/credentials
echo please input your vpn account and password
echo "ask TUD maintainer for the account and password if you don't have one"
read -p "vpn account: " vpn_account
read -p "vpn password: " vpn_password
sudo printf '%s\n' $vpn_account $vpn_password > /etc/openvpn/credentials
# TODO: use our Swarm Learning account
# GMAIL: katherlab.swarm@gmail.com
# EkFz2swarm@KATHERLAB
#sudo sed -i 's/auth-user-pass/auth-user-pass \/etc\/openvpn\/credentials/g' ./assets/openvpn_configs/tcp_files/germany1-tcp.ovpn
sudo nohup openvpn --config ./assets/openvpn_configs/tcp_files/germany1-tcp.ovpn &
