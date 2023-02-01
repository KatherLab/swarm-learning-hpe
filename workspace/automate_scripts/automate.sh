#!/bin/sh
set -eux

ip_addr=$(hostname -I | awk '{print $1}')
script_name=$(basename "${0}")
script_dir=$(realpath $(dirname "${0}"))
workspace_dir="$script_dir"/../
cd $workspace_dir

# Help function
help()
{
   echo ""
   echo "Usage: sh workspace/automate_scripts/automate.sh -c -i 192.168.33.103 -s 192.168.33.102 -w mnist-pyt-gpu"
   echo -e "\\t-w Name of the workspace module e.g. mnist-pyt-gpu, katherlab etc."
   echo -e "\\t-i Host ip address like 192.168.33.103 etc."
   echo -e "\\t-s Sentinal ip address like 192.168.33.102."
   echo -e "\\t-p Number of peers for swarm learning."
   echo -e "\\t-e Number of epochs for swarm learning."

   echo -e "\\t-a Server setup for swarm learning."
   echo -e "\\t-b Swarm learning environment setup."
   echo -e "\\t-c Final setup."
   echo -e "\\t-h Show help."
   echo ""
   exit 1
}

# Process command options
while getopts "abcw:i:s:p:e:h?" opt
do
   case "$opt" in
      w ) workspace_name="$OPTARG" ;;
      i ) host_ip="$OPTARG" ;;
      s ) sentinal_ip="$OPTARG" ;;
      p ) num_peers="$OPTARG" ;;
      e ) num_epochs="$OPTARG" ;;
      a ) ACTION=prerequisite ;;
      b ) ACTION=server_setup ;;
      c ) ACTION=final_setup ;;# sh workspace/automate_scripts/automate.sh -c -i 192.168.33.103 -w mnist-pyt-gpu -s 192.168.33.102 -p 2 -e 5
      h ) help ;;
      ? ) help ;;
   esac
done

if [ $ACTION = prerequisite ]; then
  sh ./automate_scripts/server_setup/test_open_exposed_ports.sh
  sh ./automate_scripts/server_setup/prerequisites.sh
  cd $workspace_dir
fi


if [ $ACTION = server_setup ]; then
  if [ -z "$workspace_name" ]; then
       echo "Some or all of the parameters are empty";
       help
  fi
  if [ $ip_addr = $sentinal_ip ]
    then
    sh ./automate_scripts/server_setup/server_setup.sh
  fi
  sh ./automate_scripts/server_setup/install_containers.sh
  sh ./automate_scripts/server_setup/gpu_env_setup.sh
  sh ./automate_scripts/sl_env_setup/gen_cert.sh -w "$workspace_name"
  cd $workspace_dir
fi


if [ $ACTION = final_setup ]; then
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
         sh ./automate_scripts/sl_env_setup/share_cert.sh -t "$value" -w "$workspace_name"
        done
  else
     echo "This host is not a sentinal node and will not be used for initiating the cluster, only as swarm network node"
     sn_command="--sentinel-ip=$sentinal_ip"
     sh ./automate_scripts/sl_env_setup/share_cert.sh -t "$sentinal_ip" -w "$workspace_name"

  fi
  sh ./automate_scripts/sl_env_setup/replacement.sh -w "$workspace_name" -s "$sentinal_ip" -n "$num_peers" -e "$num_epochs"
  sh ./automate_scripts/sl_env_setup/create_sldocker_wheel.sh
  sh ./automate_scripts/sl_env_setup/license_server_fix.sh
  cd $workspace_dir
fi