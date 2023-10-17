#!/usr/bin/env python3

__author__ = "Jeff"
__copyright__ = "Copyright 2023, Kather Lab"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Jeff"]
__email__ = "jiefu.zhu@tu-dresden.de"

import os
from datetime import datetime
from pathlib import Path
from categorical import categorical_aggregated_
from roc import plot_roc_curves_
from mil import deploy

#scratchDir = os.getenv('SCRATCH_DIR', '/platform/scratch')
scratchDir='/opt/hpe/swarm-learning-hpe/workspace/marugoto_mri/user/data-and-scratch/scratch/2023_04_06_155616_DUKE_transformer_swarm_learning'
#dataDir = os.getenv('DATA_DIR', '/platform/data/')
dataDir = '/mnt/sda1/Oliver'
num_epochs = int(os.getenv('MAX_EPOCHS', 64))
min_peers = int(os.getenv('MIN_PEERS', 2))
max_peers = int(os.getenv('MAX_PEERS', 7))
local_compare_flag = os.getenv('LOCAL_COMPARE_FLAG', 'False').lower() == 'true'
useAdaptiveSync = os.getenv('USE_ADAPTIVE_SYNC', 'False').lower() == 'true'
syncFrequency = int(os.getenv('SYNC_FREQUENCY', 32))
model_type = os.getenv('MODEL_TYPE', 'transformer')
current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")

data_split = 'DUKE'

#feature_dir_path = os.path.join(dataDir, data_split, 'train_val')

#test_dir = os.path.join(dataDir, data_split, 'test')
test_dir= '/mnt/sda1/Oliver/imagenet50'
out_dir = '/mnt/sda1/Oliver/att_mil_results'
#if local_compare_flag:
#    out_dir = os.path.join(scratchDir, '_'.join([str(current_time), data_split, model_type, 'local_compare']))
#else:
#    out_dir = os.path.join(scratchDir, '_'.join([str(current_time), data_split, model_type, 'swarm_learning']))

if __name__ == "__main__":


    deploy(
    clini_table = Path(os.path.join(dataDir, 'clinical_table.csv')),
    slide_csv = Path(os.path.join(dataDir, 'slide_table.csv')),
    feature_dir = Path(test_dir),
    output_path = Path(out_dir),
    model_path= Path(os.path.join('/opt/hpe/swarm-learning-hpe/workspace/marugoto_mri/user/data-and-scratch/scratch/2023_04_06_155616_DUKE_transformer_swarm_learning', 'export.pkl')),
    target_label = "Malign")
    deploy_categorical_model_(
    clini_table = Path(os.path.join(dataDir, 'clinical_table.csv')),
    slide_csv = Path(os.path.join(dataDir, 'slide_table.csv')),
    feature_dir = Path(test_dir),
    output_path = Path(out_dir)/'test',
    model_path= Path(os.path.join(out_dir, 'export_copy.pkl')),
    target_label = "Malign")

    categorical_aggregated_(os.path.join(out_dir,'patient-preds.csv'), outpath = (out_dir), target_label = "Malign")
    categorical_aggregated_(os.path.join(out_dir,'test','patient-preds.csv'), outpath = os.path.join(out_dir,'test'), target_label = "Malign")

    plot_roc_curves_([os.path.join(out_dir,'patient-preds.csv')], outpath = Path(out_dir), target_label = "Malign", true_label='1', subgroup_label=None, clini_table=None, subgroups=None)
    plot_roc_curves_([os.path.join(out_dir,'test','patient-preds.csv')], outpath = Path(out_dir)/'test', target_label = "Malign", true_label='1', subgroup_label=None, clini_table=None, subgroups=None)

    import subprocess

    # Get the container ID for the latest user-env container
    get_container_id_command = 'docker ps -a --filter "name=us*" --format "{{.ID}}" | head -n 1'
    container_id = subprocess.check_output(get_container_id_command, shell=True, text=True).strip()

    # Get the latest log for the user-env container
    get_logs_command = f"docker logs {container_id}"
    logs_process = subprocess.Popen(get_logs_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Print and log the output
    with open(os.path.join(out_dir,"container_logs.txt"), "w") as log_file:
        for line in logs_process.stdout:
            line = line.decode("utf-8").rstrip()
            print(line)
            log_file.write(line + "\n")
