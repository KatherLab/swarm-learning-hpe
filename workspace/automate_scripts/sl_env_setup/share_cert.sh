#!/bin/bash
set -eu

# Obtain IP address of current machine
ip_addr=$(ip addr show tun0 | grep 'inet ' | awk '{print $2}' | cut -f1 -d'/')

# Obtain script name and directory
script_name=$(basename "${0}")
script_dir=$(realpath $(dirname "${0}"))

# Help function
help() {
      echo ""
      echo "Usage: $script_name -u <username> -t <target_host> -w <workspace>"
      echo "Options:"
      echo " -u : The username to use when connecting to the target host."
      echo " If not specified, the default is 'swarm'."
      echo " -t : The target host to copy the certificate from."
      echo " -w : The workspace to copy the certificate to."
      echo ""
      exit 1
}

username="swarm"
# Process command options
while getopts "u:t:h?" opt; do
    case "$opt" in
      u ) username="$OPTARG" ;;
      t ) target_host="$OPTARG" ;;
      h ) help ;;
      ? ) help ;;
  esac
done

# Check if required parameters are provided
if [ -z "$target_host" ]; then
    echo "Some or all of the parameters are empty"
    help
fi
sudo apt-get -y install sshpass
# Copy the certificate from target_host to current machine's workspace
# Modify the following line to match the path to the certificate on the target host
# Modify the user name to match the user name on the target host, for example change swarm here to root

echo "First ensure you are connected with vpn with step 2 by checking with command 'hostname -I' to see if there is a IP address like 10.15.0.*"
echo "Sharing certifcates, please ask TUD maintainer for the password on sentinel host if you don't have one"
sudo scp $username@"$target_host":/opt/hpe/swarm-learning-hpe/cert/ca/capath/ca-TUD-cert.pem /opt/hpe/swarm-learning-hpe/cert/ca/capath
#sudo chmod 777 /opt/hpe/swarm-learning-hpe/cert/ca/capath/{!ca-TUD,}*"-cert.pem"
sudo scp /opt/hpe/swarm-learning-hpe/cert/ca/capath/*-cert.pem $username@"$target_host":/opt/hpe/swarm-learning-hpe/cert/ca/capath/
# If an error occurs, print an error message and exit
if [ $? -ne 0 ]; then
    echo "An error occurred while running the script. Please check the output above for more details."
    exit 1
fi
echo "Certificate copied successfully."