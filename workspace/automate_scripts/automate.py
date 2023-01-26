#!/usr/bin/env python3

__author__ = 'Jeff'
__copyright__ = 'Copyright 2023, Kather Lab'
__license__ = 'MIT'
__version__ = '0.1.0'
__maintainer__ = ['Jeff']
__email__ = 'jiefu.zhu@tu-dresden.de'

import subprocess
import os
# get the working directory
cwd = os.getcwd()
# get parrent directory
cwd = os.path.dirname(cwd)
print(cwd)
# set working directory
os.chdir(cwd)

def gpu_env_setup():
    # set gpu environment
    subprocess.call(['sh', './automate_scripts/server_setup/gpu_env_setup.sh'])

if __name__ == '__main__':
    gpu_env_setup()