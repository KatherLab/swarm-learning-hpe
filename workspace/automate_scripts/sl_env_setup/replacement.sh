#!/bin/sh
set -eux
ip_addr=$(hostname -I | awk '{print $1}')
script_name=$(basename "${0}")
script_dir=$(realpath $(dirname "${0}"))

# Process command options
while getopts "t:e:h?" opt
do
   case "$opt" in
      t ) target_host="$OPTARG" ;;
      e ) workspace="$OPTARG" ;;
   esac
done

sed -i "s+<CURRENT-PATH>+$(pwd)+g" workspace/"$workspace"/swop/swop*_profile.yaml workspace/"$workspace"/swci/taskdefs/*.yaml