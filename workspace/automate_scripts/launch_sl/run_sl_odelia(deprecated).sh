#!/bin/sh
set -eu
script_dir=$(realpath $(dirname "${0}"))
#docker rm $(docker ps -a -q)
docker rm $(docker ps --filter status=exited -q)
'''
$script_dir/../../swarm_learning_scripts/run-sl --name=sl1 --host-ip=192.168.33.102 \
--sn-ip=192.168.33.102 --sn-api-port=30304 --sl-fs-port=16000 \
--key=/opt/hpe/swarm-learning-hpe/workspace/odelia-breast-mri/cert/sl-192.168.33.102-key.pem \
--cert=/opt/hpe/swarm-learning-hpe/workspace/odelia-breast-mri/cert/sl-192.168.33.102-cert.pem \
--capath=workspace/odelia-breast-mri/cert/ca/capath --ml-it \
--ml-image=swarm_radiology_env --ml-name=ml1 \
--ml-w=/tmp/test --ml-entrypoint=python3 --ml-cmd=model/main.py \
--ml-v=workspace/odelia-breast-mri/model:/tmp/test/model \
--ml-e MODEL_DIR=model     \
--ml-e MAX_EPOCHS=5 --ml-e MIN_PEERS=2 \
--ml-e https_proxy= \
--apls-ip=192.168.33.102
'''
$script_dir/../../swarm_learning_scripts/run-sl --name=sl1 --host-ip=192.168.33.102 \
--sn-ip=192.168.33.102 --sn-api-port=30304 --sl-fs-port=16000 \
--key=/opt/hpe/swarm-learning-hpe/workspace/mnist-pyt-gpu/cert/sl-192.168.33.102-key.pem \
--cert=/opt/hpe/swarm-learning-hpe/workspace/mnist-pyt-gpu/cert/sl-192.168.33.102-cert.pem \
--capath=workspace/mnist-pyt-gpu/cert/ca/capath --ml-it \
--ml-image=swarm_radiology_env --ml-name=ml1 \
--ml-w=/tmp/test --ml-entrypoint=python3 --ml-cmd=model/main.py \
--ml-v=workspace/mnist-pyt-gpu/model:/tmp/test/model \
--ml-e MODEL_DIR=model     \
--ml-e MAX_EPOCHS=5 --ml-e MIN_PEERS=2 \
--ml-e https_proxy= \
--apls-ip=192.168.33.102