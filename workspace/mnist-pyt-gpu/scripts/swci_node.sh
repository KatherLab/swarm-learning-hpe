#!/usr/bin/env bash
./scripts/bin/run-sn -it --rm --name=sn1 --network=host-1-net --host-ip=192.168.33.102 --sentinel --sn-p2p-port=30303 --sn-api-port=30304 --key=workspace/mnist-pyt-gpu/cert/sn-1-key.pem --cert=workspace/mnist-pyt-gpu/cert/sn-1-cert.pem --capath=workspace/mnist-pyt-gpu/cert/ca/capath --apls-ip=192.168.33.102 --apls-port 5000
