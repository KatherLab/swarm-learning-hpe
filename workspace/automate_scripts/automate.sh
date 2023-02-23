#!/bin/sh
set -eux

ip_addr=$(ip addr show | awk '/inet 10\./{print $2}' | cut -d'/' -f1)
script_name=$(basename "${0}")
script_dir=$(realpath $(dirname "${0}"))

# Help function
help_old()
{
   echo ""
   echo "Example usage: sh workspace/automate_scripts/automate.sh -c -i 192.168.33.103 -s 192.168.33.102 -w mnist-pyt-gpu"
   echo -e "\\t-w Name of the workspace module e.g. mnist-pyt-gpu, katherlab etc."
   echo -e "\\t-i Host ip address like 192.168.33.103 etc."
   echo -e "\\t-s Sentinal ip address like 192.168.33.102."
   echo -e "\\t-n Number of peers for swarm learning."
   echo -e "\\t-e Number of epochs for swarm learning."

   echo -e "\\t-a Server setup for swarm learning."
   echo -e "\\t-b Swarm learning environment setup."
   echo -e "\\t-c Final setup, flage -i for this step accepts a list of target host ip address to share certificates."
   echo -e "\\t-h Show help."
   echo ""
   exit 1
}

help() {
    echo "Usage: $script_name [-a|-b|-c] [-w workspace_name] [-i host_ip] [-s sentinal_ip] [-n num_peers] [-e num_epochs] [-h]"
    echo ""
    echo "Options:"
    echo "  -a                  Run prerequisite setup steps (test_open_exposed_ports.sh and prerequisites.sh)"
    echo "  -b                  Run server setup steps (install_containers.sh, gpu_env_setup.sh, gen_cert.sh, and get_dataset.sh)"
    echo "  -c                  Run final setup steps (share_cert.sh, replacement.sh, setup_sl-cli-lib.sh, and license_server_fix.sh)"
    echo "  -w workspace_name   Set the workspace name for the distributed system"
    echo "  -i host_ip          Set the IP address of the target host"
    echo "  -s sentinal_ip      Set the IP address of the sentinel node"
    echo "  -n num_peers        Set the minimum number of peers for sync in the distributed system"
    echo "  -e num_epochs       Set the number of epochs for the machine learning model"
    echo "  -h                  Display this help message"
    echo ""
    echo "Description:"
    echo "This script automates the setup of a distributed system for machine learning. The script takes command line options and arguments to execute various steps. The steps are broken down into three actions - prerequisite, server_setup, and final_setup. The script first checks the options provided by the user, and based on that it executes the appropriate steps."
    exit 0
}

# Process command options
while getopts "abcw:i:s:n:e:h?" opt
do
   case "$opt" in
      w ) workspace_name="$OPTARG" ;;
      i ) host_ip="$OPTARG" ;;
      s ) sentinal_ip="$OPTARG" ;;
      n ) num_peers="$OPTARG" ;;
      e ) num_epochs="$OPTARG" ;;
      a ) ACTION=prerequisite ;;
      b ) ACTION=server_setup ;;
      c ) ACTION=final_setup ;;# sh workspace/automate_scripts/automate.sh -c -i 192.168.33.103 -w mnist-pyt-gpu -s 192.168.33.102 -p 2 -e 5
      h ) help ;;
      ? ) help ;;
   esac
done

if [ $ACTION = prerequisite ]; then
  sh .workspace/automate_scripts/server_setup/test_open_exposed_ports.sh
  sh .workspace/automate_scripts/server_setup/prerequisites.sh
  if [ $ip_addr = $sentinal_ip ]
    then
    sh .workspace/automate_scripts/server_setup/install_apls.sh
  fi
fi


if [ $ACTION = server_setup ]; then
  if [ -z "$workspace_name" ]; then
       echo "Some or all of the parameters are empty";
       help
  fi
  sh .workspace/automate_scripts/server_setup/install_containers.sh
  sh .workspace/automate_scripts/server_setup/gpu_env_setup.sh
  sh .workspace/automate_scripts/sl_env_setup/gen_cert.sh -w "$workspace_name"
  if [ $ip_addr != $sentinal_ip ]
    then
    sh .workspace/automate_scripts/sl_env_setup/get_dataset.sh -w "$workspace_name" -s $sentinal_ip
  fi
fi

if [ $ACTION = final_setup ]; then
  echo Please ensure the previous steps are completed on all the other hosts before running this step
  if [ -z "$host_ip" ] || [ -z "$workspace_name" ] || [ -z "$sentinal_ip" ]
    then
       echo "Some or all of the parameters are empty";
       help
    fi
  # Checks
  if [ $ip_addr = $sentinal_ip ]
  then
     echo "This host a sentinal node and will be used for initiating the cluster"
     sn_command="--sentinel"
     for value in $host_ip
        do
         echo sharing certificate with ip address $value
         sh .workspace/automate_scripts/sl_env_setup/share_cert.sh -t "$value" -w "$workspace_name"
        done
     sh .workspace/automate_scripts/sl_env_setup/license_server_fix.sh
  else
     echo "This host is not a sentinal node and will not be used for initiating the cluster, only as swarm network node"
     sn_command="--sentinel-ip=$sentinal_ip"
     #sh .workspace/automate_scripts/sl_env_setup/share_cert.sh -t "$sentinal_ip" -w "$workspace_name"

  fi
  sh .workspace/automate_scripts/sl_env_setup/replacement.sh -w "$workspace_name" -s "$sentinal_ip" -n "$num_peers" -e "$num_epochs"
  sh .workspace/automate_scripts/sl_env_setup/setup_sl-cli-lib.sh -w "$workspace_name"

fi