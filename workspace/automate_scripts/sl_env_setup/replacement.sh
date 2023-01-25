#!/bin/sh
set -eux
ip_addr=$(hostname -I | awk '{print $1}')
script_name=$(basename "${0}")
script_dir=$(realpath $(dirname "${0}"))

# Help function
help()
{
   echo ""
   echo "Ask jeff how to use the damn script"
   echo ""
   exit 1
}

# Process command options
while getopts "s:w:n:e:h?" opt
do
   case "$opt" in
      s ) sentinal_host="$OPTARG" ;;
      w ) workspace="$OPTARG" ;;
      n ) num_peers="$OPTARG" ;;
      e ) num_epochs="$OPTARG" ;;
      h ) help ;;
      ? ) help ;;
   esac
done

# Checks
if [ -z "$sentinal_host" ] || [ -z "$workspace" ] || [ -z "$num_peers" ] || [ -z "$num_epochs" ]
then
   echo "Some or all of the parameters are empty";
   help
fi

cp workspace/"$workspace"/swop/swop_profile.yaml workspace/"$workspace"/swop/swop_profile_"$ip_addr".yaml
cp workspace/"$workspace"/swci/swci-init workspace/"$workspace"/swci/swci-init_"$ip_addr"
cp workspace/"$workspace"/swci/taskdefs/swarm_task.yaml workspace/"$workspace"/swci/taskdefs/swarm_task_"$ip_addr".yaml
cp workspace/"$workspace"/swci/taskdefs/user_env_pyt_build_task.yaml workspace/"$workspace"/swci/taskdefs/user_env_pyt_build_task_"$ip_addr".yaml


sed -i "s+<CURRENT-PATH>+$(pwd)+g" workspace/"$workspace"/swop/swop_profile_"$ip_addr".yaml workspace/"$workspace"/swci/taskdefs/swarm_task_"$ip_addr".yaml workspace/"$workspace"/swci/taskdefs/user_env_pyt_build_task_"$ip_addr".yaml workspace/"$workspace"/swci/swci-init_"$ip_addr"
sed -i "s+<SN-IPADDRESS>+$sentinal_host+g" workspace/"$workspace"/swop/swop_profile_"$ip_addr".yaml workspace/"$workspace"/swci/taskdefs/swarm_task_"$ip_addr".yaml workspace/"$workspace"/swci/taskdefs/user_env_pyt_build_task_"$ip_addr".yaml workspace/"$workspace"/swci/swci-init_"$ip_addr"
sed -i "s+<HOST-IPADDRESS>+$ip_addr+g" workspace/"$workspace"/swop/swop_profile_"$ip_addr".yaml workspace/"$workspace"/swci/taskdefs/swarm_task_"$ip_addr".yaml workspace/"$workspace"/swci/taskdefs/user_env_pyt_build_task_"$ip_addr".yaml workspace/"$workspace"/swci/swci-init_"$ip_addr"
sed -i "s+<MODULE-NAME>+$workspace+g" workspace/"$workspace"/swop/swop_profile_"$ip_addr".yaml workspace/"$workspace"/swci/taskdefs/swarm_task_"$ip_addr".yaml workspace/"$workspace"/swci/taskdefs/user_env_pyt_build_task_"$ip_addr".yaml workspace/"$workspace"/swci/swci-init_"$ip_addr"
sed -i "s+<NUM-MIN_PEERS>+$num_peers+g" workspace/"$workspace"/swop/swop_profile_"$ip_addr".yaml workspace/"$workspace"/swci/taskdefs/swarm_task_"$ip_addr".yaml workspace/"$workspace"/swci/taskdefs/user_env_pyt_build_task_"$ip_addr".yaml workspace/"$workspace"/swci/swci-init_"$ip_addr"
sed -i "s+<NUM-MAX_EPOCHS>+$num_epochs+g" workspace/"$workspace"/swop/swop_profile_"$ip_addr".yaml workspace/"$workspace"/swci/taskdefs/swarm_task_"$ip_addr".yaml workspace/"$workspace"/swci/taskdefs/user_env_pyt_build_task_"$ip_addr".yaml workspace/"$workspace"/swci/swci-init_"$ip_addr"