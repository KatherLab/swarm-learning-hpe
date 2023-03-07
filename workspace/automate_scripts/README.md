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
$ sh workspace/automate_scripts/automate.sh [-a|-b|-c] [-w workspace_name] [-d host_index] [-s sentinel_ip] [-n num_peers] [-e num_epochs] [-h]"
```

## Help
To see the usage instructions, run:
```sh
$ sh workspace/automate_scripts/automate.sh -h
```


The flags are as follows:

- `[-a|-b|-c]` indicates the build stage. The script has three build stages and please ensure that you run the scripts in the correct order. The three stages are:
    - `Prerequisite`: Runs scripts that check for required software and open/exposed ports.
    - `Server setup`: Runs scripts that set up the swarm learning environment on a server.
    - `Final sl setup`: Downloads the dataset for the workspace. This will take a long time. The [-s sentinel_ip] flag is necessary. The script will download the dataset from the sentinel node.
- `-w`: Name of the workspace module, e.g. odelia-breast-mri, marugoto_mri, etc.
- `-d`: Host index, host index should be chosen from [TUD, Ribera, VHIO, Radboud, UKA, Utrecht, Mitera, Cambridge, Zurich]
- `-s`: Sentinel IP address, e.g. 10.15.0.15.
- `-p`: Minimum number of peers involved in the next training tast for swarm learning. We should coordinate this with the number of peers in the cluster.
- `-e`: Number of epochs for swarm learning. Optional. Default is 50 for marugoto_mri and 100 for odelia-breast-mri.

The script has three main actions:

- `Prerequisite`: Runs scripts that check for required software and open/exposed ports.
```sh
$ sh workspace/automate_scripts/automate.sh -a
```
- `Server setup`: Runs scripts that set up the swarm learning environment on a server.
```sh
$ sh workspace/automate_scripts/automate.sh -b -s <sentinel_ip> -d <host_index>
```
- `Download dataset`: Downloads the dataset for the workspace. This will take a long time. The [-s sentinel_ip] flag is necessary. The script will download the dataset from the sentinel node.
```sh
$ sh workspace/automate_scripts/sl_env_setup/get_dataset_scp.sh -s <sentinel_ip>
```

- `Final sl setup`: Runs scripts that finalize the setup of the swarm learning environment. Only <> is required. The [-n num_peers] and [-e num_epochs] flags are optional.
```sh
$ sh workspace/automate_scripts/automate.sh -c -w <workspace_name> -s <sentinel_ip> -d <host_index> [-n num_peers] [-e num_epochs]
```
Here is an example usage for flag -c:
```sh
$ sh workspace/automate_scripts/automate.sh -c -w odelia-breast-mri -s 10.15.0.15 -d VHIO -n 3 -e 50
```


This will run the `prerequisite`, `server_setup`, and `final_setup` actions with the specified parameters.
