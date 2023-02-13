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
from mil.helpers import (
    train_categorical_model_,
    deploy_categorical_model_,
    categorical_crossval_,
)

scratchDir = os.getenv('SCRATCH_DIR', '/platform/scratch')
dataDir = os.getenv('DATA_DIR', '/platform/data/')
current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")

data_split = '40-30-10-20'
#data_split = '25-25-25-25'

feature_dir_path = os.path.join(dataDir, data_split, 'train_val')
test_dir = os.path.join(dataDir, data_split, 'test')
out_dir = os.path.join(scratchDir, (str(current_time) + '_' +data_split+'_' + 'swarm_learning'))

if __name__ == "__main__":
    train_categorical_model_(
    clini_table = Path(os.path.join(dataDir, 'clinical_table.csv')),
    slide_csv = Path(os.path.join(dataDir, 'slide_table.csv')),
    feature_dir = Path(feature_dir_path),
    output_path = Path(out_dir),
    target_label = "Malign")

    deploy_categorical_model_(
    clini_table = Path(os.path.join(dataDir, 'clinical_table.csv')),
    slide_csv = Path(os.path.join(dataDir, 'slide_table.csv')),
    feature_dir = Path(test_dir),
    output_path = Path(out_dir),
    model_path= Path(os.path.join(out_dir, 'export.pkl')),
    target_label = "Malign")

    categorical_aggregated_(os.path.join(out_dir,'patient-preds.csv'), outpath = (out_dir), target_label = "Malign")

    plot_roc_curves_(os.path.join(out_dir,'patient-preds.csv'), outpath = (out_dir), target_label = "Malign", true_label='1')