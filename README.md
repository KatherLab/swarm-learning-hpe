# swarm-learning-hpe

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

Swarm learning learning and trails based on HPE platform

We now have a "Team Email": katherlab.swarm@gmail.com. With this we can log in to all swarm related things and use it in the code.

This repository contains:

1. SWARM Learning For Histopathology Image Analysis nad Radiology Image Analysis
2. [Work flow](https://github.com/users/Ultimate-Storm/projects/4) to help keep track of what's under process.
3. [Issue section](https://github.com/Ultimate-Storm/swarm-learning-hpe/issues) where people can dump ideas and raise
   questions encountered when using this repo.
4. Working version of mnist-pyt-gpu with automated scripts, tried on hosts 192.168.33.102 and 192.168.33.103

## Brief discription about HPE platform

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
![img.png](assets/SL structure.png)

This is the Swarm Learning framework:
![sl_node_structure.png](assets%2Fsl_node_structure.png)
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

## Install
Create a user named "swarm" and add it to the sudoers group.
Login with user "swarm" and run the following commands:

```sh
$ cd /
$ sudo mkdir opt/hpe
$ cd opt/hpe
$ sudo chmod 777 -R /opt/hpe
$ git clone https://github.com/KatherLab/swarm-learning-hpe.git
$ cd swarm-learning-hpe
$ git checkout dev_radiology
```
Requirements and dependencies will be automatically installed by the script mentioned in the following sction.

## Usage
### Setting up the Swarm Learning Environment
```sh
$ sh workspace/automate_scripts/automate.sh -<build stage> -i <target host ip to share cert> -s <sentinel host ip> -w <workspace> -n <num_peers> -e <num_epochs>"
```
Please observe this [README.md](workspace%2Fautomate_scripts%2FREADME.md) file for more details. Specific instructions are given about how to run the commands.
All the processes are automated, so you can just run the above command and wait for the process to finish.
If any problem occurs, please first try to find solution in [Troubleshooting.md](Troubleshooting.md). Then contact the maintainer of the Swarm Learning Environment and document the error in the Troubleshooting.md file.
### Running Swarm Learning Nodes
To run a Swarm Network node -> Swarm SWOP Node -> Swarm SWCI node, observe the following commands:
- To run a Swarm Network (or sentinel) node:
```sh
$ ./workspace/automate_scripts/launch_sl/run_sn.sh -s <sentinel_ip_address> -d <host_index>
```

- To run a Swarm SWOP node:
```sh
$ ./workspace/automate_scripts/launch_sl/run_swop.sh -w <workspace_name> -s <sentinel_ip_address>  -d <host_index>
```

- To run a Swarm SWCI node(SWCI node is used to generate training task runners, could be initiated by and host, but currently we suggest only the sentinel host is allowed to initiate):
```sh
$ ./workspace/automate_scripts/launch_sl/run_swci.sh -w <workspace_name> -s <sentinel_ip_address>  -d <host_index>
```

- To check the logs from training:
```sh
$ ./workspace/automate_scripts/launch_sl/check_latest_logs.sh
```


## Workflow
![Workflow.png](assets%2FWorkflow.png)
![Swarm model training protocol .png](assets%2FSwarm%20model%20training%20protocol%20.png)

## Node list
Nodes will be added to vpn and will be able to communicate with each other after setting up the Swarm Learning Environment with [Install](#install)
- Sentinel node: Dresden
  - IP address: 10.15.0.15(please put this ip address after -s flag whenever it is needed)
  - Hostname: swarm
  - Maintainer: [@Jeff](https://github.com/Ultimate-Storm)
- Other nodes: 
  - VHIO: Adrian
    - IP address: 10.15.0.16
    - Hostname: radiomics
    - Maintainer: [@Adrià](adriamarcos@vhio.net)

## Models implemented

TUD benchmarking on Duke breast mri dataset:![TUD experiments result.png](assets%2FTUD%20experiments%20result.png)
Report: [Swarm learning report.pdf](assets%2FSwarm%20learning%20report.pdf)

## Maintainers

TUD Swarm learning team

[@Jeff](https://github.com/Ultimate-Storm).


## Milestone

See this [link](https://github.com/Ultimate-Storm/swarm-learning-hpe/milestones)

## NotionPage

See this [link](https://www.notion.so/SWARM-Learning-87a7b920c88e445d81420573afb0e8ab)

## Contributing

Feel free to dive in! [Open an issue](https://github.com/Ultimate-Storm/swarm-learning-hpe/issues) or submit PRs.

Before creating a pull request, please take some time to take a look at
our [wiki page](https://github.com/Ultimate-Storm/swarm-learning-hpe/wiki), to ensure good code quality and sufficient
documentation. Don't need to follow all of the guidelines at this moment, but it would be really helpful!

### Contributors

This project exists thanks to all the people who contribute.
[@Oliver]()
[@Kevin]()


## License

[MIT](LICENSE)
