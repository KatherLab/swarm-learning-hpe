#!/bin/sh
set -eu

# Help function
help() {
   echo ""
   echo "Usage: setup_vpntunnel.sh -d <host_index> [-n]"
   echo ""
   exit 1
}
ACTION="nochange"
# Process command options
while getopts "nd:h?" opt
do
   case "$opt" in
      n ) ACTION=new ;;
      d ) host_index="$OPTARG" ;;
      h ) help ;;
      ? ) help ;;
   esac
done

if [ -z "$host_index" ]; then
   echo "Please specify your host index"
   echo "Host index should be chosen from [TUD, Ribera, VHIO, Radboud, UKA, Utrecht, Mitera, Cambridge, Zurich]"
   exit 1
fi

echo setup vpn tunnel for swarm learning
sudo apt-get -y install openvpn
if ! sudo ls /etc/openvpn/credentials >/dev/null 2>&1; then
    if [ $ACTION = new ]; then
      sudo touch /etc/openvpn/credentials
    else
      echo "Please run the script with -n option to create the credentials file"
      exit 1
    fi
    # The file exists, continue with the following commands
fi
if [ $ACTION = new ]; then
  sudo chmod 777 /etc/openvpn/credentials
  echo please input your vpn account and password
  echo "ask TUD maintainer for the account and password if you don't have one"
  read -p "vpn account: " vpn_account
  stty -echo
  read -p "vpn password: " vpn_password
  stty echo
  sudo printf '%s\n' $vpn_account $vpn_password > /etc/openvpn/credentials
fi
sudo nohup openvpn --config ./assets/openvpn_configs/good_access/$host_index.ovpn &
# If an error occurs, print an error message and exit
if [ $? -ne 0 ]; then
    echo "An error occurred while running the script. Please check the output above for more details."
    exit 1
fi
echo "VPN tunnel setup successfully."