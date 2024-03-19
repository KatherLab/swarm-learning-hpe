# swarm-learning-hpe

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

Swarm learning based on HPE platform, experiments performed based on HPE Swarm Learning version number 2.2.0

This repository contains:

1. SWARM Learning For Histopathology Image Analysis and Radiology Image Analysis
2. [Work flow](https://github.com/KatherLab/projects/4) to help keep track of what's under process.
3. [Issue section](https://github.com/KatherLab/swarm-learning-hpe/issues) where people can dump ideas and raise
   questions encountered when using this repo.
4. Working version of [marugoto_mri](workspace%2Fmarugoto_mri) for Attention MIL based model, originally suitable for histopathology images but Marta has modified it to work with MRI images.
5. Working version of [odelia-breast-mri](workspace%2Fodelia-breast-mri) for 3D-CNN model by [@Gustav](gumueller@ukaachen.de). Related instructions please refer to [ODELIA.md](ODELIA.md) and following installation steps.
6. Working version of diffusion model for generating histopathology images and Xray images, [score-based-model](workspace%2Fscore-based-model) and [swag-latent-diffusion](workspace%2Fswag-latent-diffusion). Related instructions please refer to [SWAG.md](SWAG.md) and following installation steps.

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Maintainers](#maintainers)
- [Milestone](#milestone)
- [NotionPage](#notionpage)
- [Contributing](#contributing)
- [License](#license)

## Background

### Brief description about HPE platform

Course of Swarm Leaning explained in a generally understandable
way: [https://learn.software.hpe.com/swarm-learning-essentials](https://learn.software.hpe.com/swarm-learning-essentials)

HPE Swarm Learning extends the **concept of federated learning** to **decentralized learning** by adding functionality
that **obviates the need for a central leader**. It combines the use of **AI**, **edge computing**, and **blockchain**.

HPE Swarm Learning is a **decentralized, privacy-preserving Machine Learning (ML) framework**. Swarm Learning framework
uses the computing power at, or near, the distributed data sources to run the ML algorithms that train the models. It
uses the security of a blockchain platform to share learning with peers safely and securely. In Swarm Learning, training
of the model occurs at the edge, where data is most recent, and where prompt, data-driven decisions are mostly
necessary. In this decentralized architecture, **only the insights learned are shared** with the collaborating ML peers,
not the raw data. This tremendously enhances data security and privacy.

The following image provides an overview of the Swarm Learning framework. Only the model parameters (learnings) are
exchanged between the various edge nodes and not the raw data. This ensures that the privacy of data is preserved.
![img.png](assets/SL_structure.png)

This is the Swarm Learning framework:
![sl_node_structure.png](assets%2Fsl_node_structure.png)

## Install

### Prerequisites
#### Hardware recommendations
* 64 GB of RAM (32 GB is the absolute minimum)
* 16 CPU cores (8 is the absolute minimum)
* an NVIDIA GPU with 48 GB of RAM (24 is the minimum)
* 8 TB of Storage (4 TB is the absolute minimum)
* We deliberately want to show that we can work with lightweight hardware like this. Here are three quotes for systems like this for less than 10k EUR (Lambda, Dell Precision, and Dell Alienware)
* Typical installation time can take 30 minutes to build up the necessary dependencies and another 1 hour to build up the environment for running demo experiments
  
#### Operating System
* Ubuntu 20.04 LTS
  * We have tested the Swarm Learning Environment on [Ubuntu 20.04 LTS, Ubuntu 22.04.2 LTS, Ubuntu 20.04.5 LTS] and they work fine. 
  *  Any experimental release of Ubuntu greater than LTS 20.04 MAY result in unsuccessful swop node running.
  * It also works on WSL2(Ubuntu 20.04.2 LTS) on Windows systems. WSL1 may have some issues with the docker service.

### Upgrade the Swarm Learning Environment from Older Version
1. Run the following command to upgrade the Swarm Learning Environment from 1.x.x to 2.x.x
```sh
sh workspace/automate_scripts/server_setup/cleanup_old_sl.sh
```
Then proceed to 1. `Prerequisite` in [Setting up the Swarm Learning Environment](#setting-up-the-swarm-learning-environment)

### Setting up the user and repository
1. Create a user named "swarm" and add it to the sudoers group.
Login with user "swarm".
```sh
sudo adduser swarm
sudo usermod -aG sudo swarm
sudo su - swarm
```
2. Add the Docker user to the sudoers group
```sh
sudo usermod -aG docker swarm
```
After running this command, you will need to log out and log back in for the changes to take effect, or you can use the newgrp command like so:
```sh
newgrp docker
```

3. Run the following commands to set up the repository:

```sh
cd / && sudo mkdir opt/hpe && cd opt/hpe && sudo chmod 777 -R /opt/hpe
git clone https://github.com/KatherLab/swarm-learning-hpe.git && cd swarm-learning-hpe
```

4. Install cuda environment and nvidia drivers, as soon as you could see correct outputs of the following command you may proceed.
```sh
nvidia-smi
```
Please disable secure boot. On some systems, Secure Boot might prevent unsigned kernel modules (like NVIDIA's) from loading.
Check Loaded Kernel Modules:
- To see if the NVIDIA kernel module is loaded:
```sh
lsmod | grep nvidia
```
Review System Logs:
- Sometimes, system logs can provide insights into any issues with the GPU or driver:
```sh
dmesg | grep -i nvidia
```
Manually Load the NVIDIA Module:
- You can try manually loading the NVIDIA kernel module using the modprobe command:
```sh
sudo modprobe nvidia
```
Requirements and dependencies will be automatically installed by the script mentioned in the following section.

### Decide on the connection method
Determine how your server will connect to the network. You have the option to configure the network interface for a VPN connection using GoodAccess, direct Ethernet connection, or through Tailscale. Follow the instructions below to set up the network interface based on your choice:

- For a VPN connection using GoodAccess(For Odelia project, this is the recommended method.)
sh /workspace/automate_scripts/server_setup/replace_network_interface.sh --goodaccess

- For a direct Ethernet connection(For proof of concept, this is the recommended method.)
sh /workspace/automate_scripts/server_setup/replace_network_interface.sh --local

- For a Tailscale connection(For Swag project, this is the recommended method.)
sh /workspace/automate_scripts/server_setup/replace_network_interface.sh --tailscale


### Setting up the Swarm Learning Environment
**PLEASE REPLACE THE `<PLACEHOLDER>` WITH THE CORRESPONDING VALUE!**

`<sentinel_ip>` = `192.168.33.100` currently it's the IP assigned by VPN server for TUD host.

`<host_index>` = Your institute's name. For ODELIA project should be chosen from `TUD` `Ribera` `VHIO` `Radboud` `UKA` `Utrecht` `Mitera` `Cambridge` `Zurich`

`<workspace_name>` = The name of the workspace you want to work on. You can find available modules under `workspace/` folder. Each module corresonds to a radiology model. Currently we suggest to use `odelia-breast-mri` or `marogoto_mri` here.

**Please only proceed to the next step by observing "... is done successfully" from the log**

0. Optional: download preprocessed datasets. Please refer to the [Data Preparation](ODELIA) section for more details.

1. `Prerequisite`: Runs scripts that check for required software and open/exposed ports.
```sh
sh workspace/automate_scripts/automate.sh -a
```
2. `Server setup`: Runs scripts that set up the swarm learning environment on a server.
```sh
sh workspace/automate_scripts/automate.sh -b -s <sentinel_ip> -d <host_index>
```
3. `Final setup`: Runs scripts that finalize the setup of the swarm learning environment. Only <> is required. The [-n num_peers] and [-e num_epochs] flags are optional.
```sh
sh workspace/automate_scripts/automate.sh -c -w <workspace_name> -s <sentinel_ip> -d <host_index> [-n num_peers] [-e num_epochs]
```

Optional 5. Reconnect to VPN
```sh
sh /workspace/automate_scripts/server_setup/setup_vpntunnel.sh
```

In case your machine got restarted or lost the vpn connection. Here is the guide to reconnect: [VPN connect guide](https://support.goodaccess.com/configuration-guides/linux)
The file.ovpn is the config file that TUD assigned to you.

If a problem is encountered, please observe this [README.md](workspace%2Fautomate_scripts%2FREADME.md) file for step-by-step setup. Specific instructions are given about how to run the commands.
All the processes are automated, so you can just run the above command and wait for the process to finish.

If any problem occurs, please first try to figure out which step is going wrong, try to google for solutions and find solution in [Troubleshooting.md](Troubleshooting.md). Then contact the maintainer of the Swarm Learning Environment and document the error in the Troubleshooting.md file.


Following these structured steps will help in maintaining a well-organized dataset, thereby enhancing data management and processing in your projects.



### Running Swarm Learning Nodes
To run a Swarm Network node -> Swarm SWOP Node -> Swarm SWCI node. Please open a terminal for each of the nodes to run. Observe the following commands:
#### SN
- To run a Swarm Network (or sentinel) node:
```sh
./workspace/automate_scripts/launch_sl/run_sn.sh -s <sentinel_ip_address> -d <host_index>
```
or
```sh
runsn
```
#### SWOP
- To run a Swarm SWOP node:
```sh
./workspace/automate_scripts/launch_sl/run_swop.sh -w <workspace_name> -s <sentinel_ip_address>  -d <host_index>
```
or
```sh
runswop
```
#### SWCI

- To run a Swarm SWCI node(SWCI node is used to generate training task runners, could be initiated by any host, but currently we suggest only the sentinel host is allowed to initiate):
```sh
./workspace/automate_scripts/launch_sl/run_swci.sh -w <workspace_name> -s <sentinel_ip_address>  -d <host_index>
```
or
```sh
runswci
```


- To check the logs from training:
```sh
./workspace/automate_scripts/launch_sl/check_latest_log.sh
```
or
```sh
cklog [--ml] [--swci] [--swop] [--sn] 
```


- To stop the Swarm Learning nodes, --[node_type] is optional, if not specified, all the nodes will be stopped. Otherwise, could specify --sn, --swop for example.
```sh
./workspace/swarm_learning_scripts/stop-swarm --[node_type]
```
or
```sh
stopswarm [--node_type]
```

- To view results, see logs under `workspace/<workspace_name>/user/data-and-scratch/scratch`

Please observe [Troubleshooting.md](Troubleshooting.md) section 10 for successful running logs of swop and sn nodes. The process will keep running and displaying logs so you will need to open a new terminal to run other commands.
* Typical run time can take 3 hours for experiments trained on the DUKE dataset with ResNet50-3D with three nodes involved.


## Workflow

![Workflow.png](assets%2FWorkflow.png)
![Swarm model training protocol .png](assets%2FSwarm%20model%20training%20protocol%20.png)



## Models implemented

TUD benchmarking on Duke breast mri dataset:![TUD experiments result.png](assets%2FTUD%20experiments%20result.png)

Report: [Swarm learning report.pdf](assets%2FSwarm%20learning%20report.pdf)

## Maintainers

TUD Swarm learning team

[@Jeff](https://github.com/Ultimate-Storm).

Wanna a 24-hours support? Configure your TeamViewer with the following steps and contact me through slack. Thanks [@Adrià](adriamarcos@vhio.net) for instructions.

1. Enable remote control in the ubuntu settings
![ubuntu_remote_control.png](assets%2Fubuntu_remote_control.png)
2. Install TeamViewer and login with username: `adriamarcos@vhio.net` and password: `2wuHih4qC5tEREM`
3. Add the computer to the account ones, so that it can be controlled. You have it here: [link](https://community.teamviewer.com/English/kb/articles/4464-assign-a-device-to-your-accountuthorize)![TV add device.png](assets%2FTV%20add%20device.png)
4. I'd advise you to set the computer to never enter the sleeping mode or darken the screen just in case. Also, if you want to use different users remember this has to be done in all them and the TV session need to be signed in all them as well.

## Milestone

See this [link](https://github.com/KatherLab/swarm-learning-hpe/milestones)

## NotionPage

See this [link](https://www.notion.so/SWARM-Learning-87a7b920c88e445d81420573afb0e8ab)

## Contributing

Feel free to dive in! [Open an issue](https://github.com/KatherLab/swarm-learning-hpe/issues) or submit PRs.

Before creating a pull request, please take some time to take a look at
our [wiki page](https://github.com/KatherLab/swarm-learning-hpe/wiki), to ensure good code quality and sufficient
documentation. Don't need to follow all of the guidelines at this moment, but it would be really helpful!

### Contributors

This project exists thanks to all the people who contribute.
[@Oliver](https://github.com/oliversaldanha25)
[@Kevin](https://github.com/kevinxpfeiffer)

## Credits

This project uses platform from the following repositories:

- [HewlettPackard/swarm-learning](https://github.com/HewlettPackard/swarm-learning): Created by [HewlettPackard](https://github.com/HewlettPackard)

## License

[MIT](LICENSE)
