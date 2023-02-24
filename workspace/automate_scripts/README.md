# Automation Scripts

This repository contains automation scripts for setting up a swarm learning environment on a cluster of machines.

## Contents
This script automates the process of setting up a workspace for swarm learning. It puts together several other scripts to perform the following tasks:

- Check for open and exposed ports on the host server
- Install prerequisites for the workspace
- Install necessary containers for the workspace
- Set up GPU environment for the workspace
- Generate and share SSL certificate for the workspace
- Set up swarm learning environment
- Set up swarm network node and/or sentinel node
- Share the SSL certificate with other nodes in the cluster
- Configure swarm learning environment for the workspace

## Usage

To use the scripts, clone this repository onto your machine and run `automate.sh` with the required flags:

```sh
$ sh workspace/automate_scripts/automate.sh -<build stage> -i <target host ip to share cert> -s <sentinal host ip> -w <workspace> -n <num_peers> -e <num_epochs>"
```

## Help
To see the usage instructions, run:
```sh
$ sh workspace/automate_scripts/automate.sh -h
```


The flags are as follows:

- `-w`: Name of the workspace module, e.g. mnist-pyt-gpu, katherlab, etc.
- `-i`: Host IP address, e.g. 192.168.33.103.
- `-s`: Sentinel IP address, e.g. 192.168.33.102.
- `-p`: Number of peers for swarm learning.
- `-e`: Number of epochs for swarm learning.

The script has three main actions:

- `prerequisite`: Runs scripts that check for required software and open/exposed ports.
```sh
$ sh workspace/automate_scripts/automate.sh -a
```
- `server_setup`: Runs scripts that set up the swarm learning environment on a server.
```sh
$ sh workspace/automate_scripts/automate.sh -b -w <workspace_name> -s <sentinal_ip>
```
- `final_setup`: Runs scripts that finalize the setup of the swarm learning environment. The [-i list_of_ip] flag is necessary only for sentinal node. If it is specified, the script will share the SSL certificate with the nodes listed in the cluster.
```sh
$ sh workspace/automate_scripts/automate.sh -c -w <workspace_name> -s <sentinal_ip> [-n num_peers] [-e num_epochs] [-i list_of_ip]
```
Here is an example usage for flag -c:
```sh
$ sh workspace/automate_scripts/automate.sh -c -i 192.168.33.103 -w mnist-pyt-gpu -s 192.168.33.102 -p 2 -e 5
```


This will run the `prerequisite`, `server_setup`, and `final_setup` actions with the specified parameters.
