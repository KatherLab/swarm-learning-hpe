#!/bin/sh
set -eu

# Define script name and directory
script_name=$(basename "${0}")
script_dir=$(realpath $(dirname "${0}"))

# Define usage function
usage() {
  echo "Usage: $script_name [-h]"
  echo "Test whether required ports are open for HPE Swarm Learning"
  echo ""
  echo "Optional arguments:"
  echo "  -h, --help       Show this help message and exit"
  exit 1
}

# Process command-line arguments
while [ $# -gt 0 ]; do
  case $1 in
    -h|--help)
      usage
      ;;
    *)
      echo "Invalid argument: $1"
      usage
      ;;
  esac
done

# Define test_port function
test_port() {
  port=$1
  if (sudo iptables -L -n | grep $port | grep tcp); then
    echo "Port $port: open"
  else
    echo "Port $port: closed"
    echo "Port $port: opening..."
    open_port $port
  fi
}

# Define open_port function
open_port() {
  port=$1
  sudo iptables -A INPUT -p tcp --dport $1 -j ACCEPT
  sudo iptables -A OUTPUT -p tcp --dport $1 -j ACCEPT
  sudo iptables -A FORWARD -p tcp --dport $1 -j ACCEPT
  test_port $1
}

# Test standard HPE Swarm Learning ports
test_port 22
test_port 5814
#test_port 5000
test_port 30303
test_port 30304
test_port 30305
test_port 30306

# Test use case pending ports
test_port 16000
test_port 17000
test_port 18000
test_port 19000

# If an error occurs, print an error message and exit
if [ $? -ne 0 ]; then
    echo "An error occurred while running the script. Please check the output above for more details."
    exit 1
fi
echo "All required ports are open."