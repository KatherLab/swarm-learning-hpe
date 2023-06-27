#!/usr/bin/env python3

__author__ = "Jeff"
__copyright__ = "Copyright 2023, Kather Lab"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Jeff"]
__email__ = "jiefu.zhu@tu-dresden.de"
import os
import glob
import re  # import regular expressions module

# initialize the dictionary
auc_dict = {}

# define the base directory
base_dir = '/mnt/sda1/Duke Compare/ext_val_results'

# get all directories
all_dirs = [x[0] for x in os.walk(base_dir)]

for dir in all_dirs:
    # break down the directory path
    dir_parts = dir.split(os.sep)
    #print(dir_parts)
    if len(dir_parts) > 8:   # updated here to ensure we have experiment type in the path
        host = dir_parts[-4]
        model = dir_parts[-1].split('_')[-3]
        experiment_type = '_'.join(dir_parts[-1].split('_')[-2:])  # new variable for the experiment type
        auc_files = glob.glob(f'{dir}/AUC.txt')

        for file in auc_files:
            with open(file, 'r') as f:
                line = f.readline().strip()  # read the line
                match = re.search(r"AUC = (.+?) \$\\pm\$", line)  # search for AUC score
                if match is not None:
                    auc = match.group(1)  # extract AUC score


            if experiment_type == 'local_compare':
                if host == 'Host_100':
                    epxtpye = 'Localhost3 - 10% data'
                elif host == 'Host_101':
                    epxtpye = 'Localhost2 - 30% data'
                elif host == 'Host_Sentinal':
                    epxtpye = 'Localhost1 - 40% data'
            elif experiment_type == 'swarm_learning':
                epxtpye = 'swarm_learning'

            auc_dict[model][epxtpye].append(auc)
    elif len(dir_parts) >= 8 and 'Host_100' not in dir_parts:
        host = dir_parts[-3]
        model = dir_parts[-1].split('_')[-3]
        experiment_type = '_'.join(dir_parts[-1].split('_')[-2:])  # new variable for the experiment type
        auc_files = glob.glob(f'{dir}/AUC.txt')
        print(dir_parts)
        for file in auc_files:
            with open(file, 'r') as f:
                line = f.readline().strip()  # read the line
                match = re.search(r"AUC = (.+?) \$\\pm\$", line)  # search for AUC score
                if match is not None:
                    auc = match.group(1)  # extract AUC score

            # create nested dictionaries if they do not exist
            if host not in auc_dict:
                auc_dict[host] = {}
            if model not in auc_dict[host]:
                auc_dict[host][model] = {}
            if experiment_type not in auc_dict[host][model]:  # new level for experiment type
                auc_dict[host][model][experiment_type] = []
            print(host)
            # append auc value
            if experiment_type == 'local_compare':
                if host == 'Host_100':
                    epxtpye = 'Localhost3 - 10% data'
                elif host == 'Host_101':
                    epxtpye = 'Localhost2 - 30% data'
                elif host == 'Host_Sentinal':
                    epxtpye = 'Localhost1 - 40% data'
            elif experiment_type == 'swarm_learning':
                epxtpye = 'swarm_learning'
            auc_dict[model][epxtpye].append(auc)
            #print(auc_dict)
print(auc_dict)
