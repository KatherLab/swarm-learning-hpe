#!/bin/sh
set -eux
script_dir=$(realpath $(dirname "${0}"))
#docker rm $(docker ps -a -q)
docker rm $(docker ps --filter status=exited -q)

# Help function
help()
{
   echo ""
   echo "Ask jeff how to use the damn script"
   echo ""
   exit 1
}

# Process command options
while getopts "w:h?" opt
do
   case "$opt" in
      w ) workspace="$OPTARG" ;;
      #i ) host="$OPTARG" ;;
      #s ) sentinal="$OPTARG" ;;
      h ) help ;;#sh workspace/automate_scripts/launch_sl/run_sn.sh -w mnist-pyt-gpu -s 192.168.33.102
      ? ) help ;;
   esac
done


$script_dir/../../swarm_learning_scripts/run-sl --name=sl1 --host-ip=192.168.33.102 \
--sn-ip=192.168.33.102 --sn-api-port=30304 --sl-fs-port=16000 \
--key=/opt/hpe/swarm-learning-hpe/workspace/"$workspace"/cert/sl-192.168.33.102-key.pem \
--cert=/opt/hpe/swarm-learning-hpe/workspace/"$workspace"/cert/sl-192.168.33.102-cert.pem \
--capath=workspace/"$workspace"/cert/ca/capath --ml-it \
--ml-image=swarm --ml-name=ml1 \
--ml-w=/tmp/test --ml-entrypoint=python3 --ml-cmd=model/main.py \
--ml-v=workspace/"$workspace"/model:/tmp/test/model \
--ml-e MODEL_DIR=model     \
--ml-e MAX_EPOCHS=5 --ml-e MIN_PEERS=2 \
--ml-e https_proxy= \
--apls-ip=192.168.33.102
