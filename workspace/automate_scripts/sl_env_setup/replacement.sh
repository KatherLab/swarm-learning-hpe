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
while getopts "t:w:h?" opt
do
   case "$opt" in
      s ) sentinal_host="$OPTARG" ;;
      w ) workspace="$OPTARG" ;;
      h ) help ;;
      ? ) help ;;
   esac
done

# Checks
if [ -z "$sentinal_host" ] || [ -z "$workspace" ]
then
   echo "Some or all of the parameters are empty";
   help
fi

cp workspace/"$workspace"/swop/swop_profile.yaml workspace/"$workspace"/swop/swop_profile_"$ip_addr".yaml
cp workspace/"$workspace"/swci/swci_init workspace/"$workspace"/swci/swci_init_"$ip_addr"
cp workspace/"$workspace"/swci/taskdefs/*.yaml workspace/"$workspace"/swci/taskdefs/*_"$ip_addr".yaml

sed -i "s+<CURRENT-PATH>+$(pwd)+g" workspace/"$workspace"/swop/swop_profile.yaml workspace/"$workspace"/swci/taskdefs/*.yaml workspace/"$workspace"/swci/swci_init
sed -i "s+<SN-IPADDRESS>+$(sentinal_host)+g" workspace/"$workspace"/swop/swop_profile.yaml workspace/"$workspace"/swci/taskdefs/*.yaml workspace/"$workspace"/swci/swci_init
sed -i "s+<HOST-IPADDRESS>+$(ip_addr)+g" workspace/"$workspace"/swop/swop_profile.yaml workspace/"$workspace"/swci/taskdefs/*.yaml workspace/"$workspace"/swci/swci_init
sed -i "s+<MODULE-NAME>+$(workspace)+g" workspace/"$workspace"/swop/swop_profile.yaml workspace/"$workspace"/swci/taskdefs/*.yaml workspace/"$workspace"/swci/swci_init