#!/bin/sh
set -eu
script_name=$(basename "${0}")

# Help function
help_old()
{
   echo ""
   echo "Example usage: sh workspace/automate_scripts/automate.sh -c -s 192.168.33.102 -w mnist-pyt-gpu"
   echo -e "\\t-w Name of the workspace module e.g. mnist-pyt-gpu, katherlab etc."
   #echo -e "\\t-i Host ip address like 192.168.33.103 etc."
   echo -e "\\t-s Sentinel ip address like 192.168.33.102."
   echo -e "\\t-n Number of peers for swarm learning."
   echo -e "\\t-e Number of epochs for swarm learning."

   echo -e "\\t-a Server setup for swarm learning."
   echo -e "\\t-b Swarm learning environment setup."
   echo -e "\\t-c Final setup, flag -i for this step accepts a list of target host ip address to share certificates."
   echo -e "\\t-h Show help."
   echo ""
   exit 1
}

help() {
    echo "Usage: $script_name [-a|-b|-c] [-w workspace_name] [-d host_index] [-s sentinel_ip] [-n num_peers] [-e num_epochs] [-h]"
    echo ""
    echo "Options:"
    echo "  -a                  Run prerequisite setup steps (test_open_exposed_ports.sh and prerequisites.sh)"
    echo "  -b                  Run server setup steps (install_containers.sh, gpu_env_setup.sh, gen_cert.sh, and get_dataset_scp.sh)"
    echo "  -c                  Run final setup steps (share_cert.sh, replacement.sh, setup_sl-cli-lib.sh, and license_server_fix.sh)"
    echo "  -w workspace_name   Set the workspace name for the distributed system"
    #echo "  -i host_ip          Set the IP address of the target host"
    echo "  -d host_index       Set the host index for generating SSL certificates"
    echo "  -s sentinel_ip      Set the IP address of the sentinel node"
    echo "  -n num_peers        Set the minimum number of peers for sync in the distributed system"
    echo "  -e num_epochs       Set the number of epochs for the machine learning model"
    echo "  -h                  Display this help message"
    echo "  --debug             Enable debugging mode (enables error checking, unset variable checking, and debugging output)"
    echo ""
    echo "Description:"
    echo "This script automates the setup of a distributed system for machine learning. The script takes command line options and arguments to execute various steps. The steps are broken down into three actions - prerequisite, server_setup, and final_setup. The script first checks the options provided by the user, and based on that it executes the appropriate steps."
    exit 0
}
num_peers=2
num_epochs=100
# Process command options
while getopts "abcw:i:d:s:n:e:h?" opt
do
   case "$opt" in
      w ) workspace_name="$OPTARG" ;;
      #i ) host_ip="$OPTARG" ;;
      d ) host_index="$OPTARG" ;;
      s ) sentinel_ip="$OPTARG" ;;
      n ) num_peers="$OPTARG" ;;
      e ) num_epochs="$OPTARG" ;;
      a ) ACTION=prerequisite ;;
      b ) ACTION=server_setup ;;
      c ) ACTION=final_setup ;;# sh workspace/automate_scripts/automate.sh -c -i 192.168.33.103 -w mnist-pyt-gpu -s 192.168.33.102 -p 2 -e 5
      h ) help ;;
      --debug ) set -eux ;;  # Enable debugging mode
      ? ) help ;;
   esac
done

if [ $ACTION = prerequisite ]; then

  sh ./workspace/automate_scripts/server_setup/prerequisites.sh
  #if [ -z "$sentinel" ]; then
  #echo "Specified sentinel node address, will install apls server"
  #fi
  echo "Prerequisite setup steps completed successfully. Please proceed to the next step."
fi


if [ $ACTION = server_setup ]; then
  if [ -z "$host_index" ]; then
       echo "Please specify your host index"
       echo "Host index should be chosen from [TUD, Ribera, VHIO, Radboud, UKA, Utrecht, Mitera, Cambridge, Zurich]"
  fi
    if [ -z "$sentinel_ip" ];
    then
       echo "sentinel_ip required"
       help
  fi
  sh ./workspace/automate_scripts/server_setup/install_containers.sh
  sh ./workspace/automate_scripts/server_setup/gpu_env_setup.sh
  sh ./workspace/automate_scripts/sl_env_setup/gen_cert.sh -i "$host_index"
  sh ./workspace/automate_scripts/sl_env_setup/setup_sl-cli-lib.sh
  sudo sh ./workspace/automate_scripts/server_setup/setup_vpntunnel.sh -d "$host_index" -n
  echo waiting for VPN to get connected
  sleep 10
  ip_addr=$(ip addr show tun0 2>/dev/null | grep 'inet ' | awk '{print $2}' | cut -f1 -d'/')

if [ -z "$ip_addr" ]; then
    echo "Error: tun0 interface not found. Please connect to the VPN first. Use script setup_vpntunnel.sh"
    exit 1
fi
  if [ $ip_addr = $sentinel_ip ]
      then
      sudo sh ./workspace/automate_scripts/server_setup/install_apls.sh
  fi
  sh ./workspace/automate_scripts/server_setup/test_open_exposed_ports.sh
  if [ $ip_addr != $sentinel_ip ]
    then
      echo "please download the dataset from the sentinel node if needed, this will take some time"
      echo "run with either command, get_dataset_gdown without vpn, get_dataset_scp with vpn"
      echo "sh ./workspace/automate_scripts/sl_env_setup/get_dataset_gdown.sh"
      echo "sh ./workspace/automate_scripts/sl_env_setup/get_dataset_scp.sh -s $sentinel_ip"
    #sh ./workspace/automate_scripts/sl_env_setup/get_dataset_scp.sh -w "$workspace_name" -s $sentinel_ip
  fi
  echo "Server setup steps completed successfully. Please proceed to final setup."
fi

if [ $ACTION = final_setup ]; then
  if [ -z "$workspace_name" ] || [ -z "$sentinel_ip" ] || [ -z "$host_index" ];
    then
       echo "workspace_name and sentinel_ip are required"
       help
  fi

  echo Please ensure the previous steps are completed on all the other hosts before running this step
 ip_addr=$(ip addr show tun0 2>/dev/null | grep 'inet ' | awk '{print $2}' | cut -f1 -d'/')
  if [ $ip_addr = $sentinel_ip ]
    then
      echo "This host a sentinel node and will skip certs sharing"
    else
      echo "This host is not a sentinel node and will share certs with sentinel node"
      sh ./workspace/automate_scripts/sl_env_setup/share_cert.sh -t "$sentinel_ip"
  fi
if [ -z "$ip_addr" ]; then
    echo "Error: tun0 interface not found. Please connect to the VPN first. Use script setup_vpntunnel.sh"
    exit 1
fi
  # Checks
  if [ $ip_addr = $sentinel_ip ]
  then
     echo "This host a sentinel node and will be used for initiating the cluster"
     sn_command="--sentinel"
     #for value in $host_ip
        #do
         #echo sharing certificate with ip address $value
         #sh ./workspace/automate_scripts/sl_env_setup/share_cert.sh -t "$value" -w "$workspace_name"
        #done
     sh ./workspace/automate_scripts/sl_env_setup/license_server_fix.sh
  else
     echo "This host is not a sentinel node and will not be used for initiating the cluster, only as swarm network node"
     #sn_command="--sentinel-ip=$sentinel_ip"
     #sh ./workspace/automate_scripts/sl_env_setup/share_cert.sh -t "$sentinel_ip" -w "$workspace_name"

  fi
  # set default values if num_peers or num_epochs not specified

  sh ./workspace/automate_scripts/sl_env_setup/replacement.sh -w "$workspace_name" -s "$sentinel_ip" -n "$num_peers" -e "$num_epochs" -d "$host_index"
  echo "Final setup steps completed successfully. Please proceed to the next step for running Swarm Learning nodes."
  echo "Adding alias for quick run"
  sh ./workspace/automate_scripts/sl_env_setup/setup_aliases.sh -d "$host_index"
fi

# If an error occurs, print an error message and exit
if [ $? -ne 0 ]; then
    echo "An error occurred while running the script. Please check the output above for more details."
    exit 1
fi
