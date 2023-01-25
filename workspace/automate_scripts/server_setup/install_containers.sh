#!/bin/sh
set -eux

echo "Login to hub.myenterpriselicense.hpe.com"
sudo docker login -u jiefu.zhu@tu-dresden.de -p hpe_eval hub.myenterpriselicense.hpe.com
# sudo docker login -u kevin.pfeiffer@tu-dresden.de -p hpe_eval hub.myenterpriselicense.hpe.com

echo "Download Swarm Network (SN) Node"
sudo docker pull hub.myenterpriselicense.hpe.com/hpe_eval/swarm-learning/sn:1.2.0

echo "Download Swarm Learning (SL) Node"
sudo docker pull hub.myenterpriselicense.hpe.com/hpe_eval/swarm-learning/sl:1.2.0

echo "Download Swarm Learning Command Interface (SWCI) Node"
sudo docker pull hub.myenterpriselicense.hpe.com/hpe_eval/swarm-learning/swci:1.2.0

echo "Download Swarm Operator (SWOP) Node"
sudo docker pull hub.myenterpriselicense.hpe.com/hpe/swarm-learning/swop:1.2.0

cd ./license_and_softwares/
sudo tar -xf HPE_SWARM_LEARNING_DOCS_EXAMPLES_SCRIPTS_Q2V41-11033.tar.gz -C /opt/hpe/swarm-learning-hpe/