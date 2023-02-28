#!/usr/bin/env python3

__author__ = "Jeff"
__copyright__ = "Copyright 2023, Kather Lab"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Jeff"]
__email__ = "jiefu.zhu@tu-dresden.de"

import subprocess

workspace_name = input("Enter workspace name: ")
sentinel_ip = input("Enter Sentinel IP address: ")

# Launch run_sn.sh script in a new terminal window
subprocess.Popen(['gnome-terminal', '-e', f'./workspace/automate_scripts/launch_sl/run_sn.sh -w {workspace_name} -s {sentinel_ip}'])

# Launch run_swop.sh script in a new terminal window
subprocess.Popen(['gnome-terminal', '-e', f'./workspace/automate_scripts/launch_sl/run_swop.sh -w {workspace_name} -s {sentinel_ip}'])

# Launch run_swci.sh script in a new terminal window
subprocess.Popen(['gnome-terminal', '-e', f'./workspace/automate_scripts/launch_sl/run_swci.sh -w {workspace_name} -s {sentinel_ip}'])
