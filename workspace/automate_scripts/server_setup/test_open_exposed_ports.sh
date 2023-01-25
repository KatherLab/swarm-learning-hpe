#!/bin/sh
set -eu

script_name=$(basename "${0}")
script_dir=$(realpath $(dirname "${0}"))

# help function
help()
{
   echo ""
   echo "Ask kevin how to use the damn script"
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

# checks ports
test_port(){
port=$1
if (sudo iptables -L -n | grep $port | grep tcp); then
	echo "Port $port: open"
else
	echo "Port $port: closed"
	echo "Port $port: will be opened"
	open_port $port
fi
}

# open
open_port(){
port=$1
sudo iptables -A INPUT -p tcp --dport $1 -j ACCEPT
sudo iptables -A OUTPUT -p tcp --dport $1 -j ACCEPT
sudo iptables -A FORWARD -p tcp --dport $1 -j ACCEPT
test_port $1
}

# standard HPE Swarm Learing ports
test_port 22
test_port 5814
test_port 5000
test_port 30303
test_port 30304
test_port 30305
test_port 30306

# Use case pending ports
test_port 16000
test_port 17000
test_port 18000
test_port 19000
