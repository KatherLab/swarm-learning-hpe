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
    sudo printf '%s\n' 'jeffzhu6969@gmail.com' '5885ohdude' > /etc/openvpn/credentials
fi

# TODO: use our Swarm Learning account
# GMAIL: katherlab.swarm@gmail.com
# EkFz2swarm@KATHERLAB
sudo sed -i 's/auth-user-pass/auth-user-pass \/etc\/openvpn\/credentials/g' ./assets/openvpn_configs/tcp_files/germany1-tcp.ovpn
sudo nohup openvpn --config ./assets/openvpn_configs/tcp_files/germany1-tcp.ovpn &
