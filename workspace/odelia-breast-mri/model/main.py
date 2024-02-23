from datetime import datetime
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import os
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torch.utils.data.dataset import Subset
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from data.datamodules import DataModule
from utils.roc_curve import plot_roc_curve, cm2acc, cm2x
import monai.networks.nets as nets
import torch
from swarmlearning.pyt import SwarmCallback
from pytorch_lightning.callbacks import Callback
from model_selector import select_model
from env_config import *
from collections import Counter

class User_swarm_callback(Callback):
    def __init__(self, swarmCallback):
        self.swarmCallback = swarmCallback

    #def on_train_start(self, trainer, pl_module):
    #    self.swarmCallback.on_train_begin()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.swarmCallback.on_batch_end()

    def on_train_epoch_end(self, trainer, pl_module):
        self.swarmCallback.on_epoch_end()

    #def on_train_end(self, trainer, pl_module):
    #    self.swarmCallback.on_train_end()


if __name__ == "__main__":
    env_vars = load_environment_variables()
    print('model_name: ', env_vars['model_name'])

    predict, predict_last = load_prediction_modules(env_vars['prediction_flag'])
    ds = prepare_dataset(env_vars['task_data_name'], env_vars['data_dir'])
    path_run_dir = generate_run_directory(env_vars['scratch_dir'], env_vars['task_data_name'], env_vars['model_name'], env_vars['local_compare_flag'])

    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    print(f"Using {accelerator} for training")

    local_compare_flag = env_vars['local_compare_flag']
    max_epochs = env_vars['max_epochs']
    min_peers = env_vars['min_peers']
    max_peers = env_vars['max_peers']
    dataDir = env_vars['data_dir']
    task_data_name = env_vars['task_data_name']

    labels = ds.get_labels()

    # Generate indices and perform stratified split
    indices = list(range(len(ds)))
    train_indices, val_indices = train_test_split(indices, test_size=0.2, stratify=labels, random_state=42)

    # Create training and validation subsets
    ds_train = Subset(ds, train_indices)
    ds_val = Subset(ds, val_indices)
    # Extract training labels using the train_indices
    train_labels = [labels[i] for i in train_indices]
    # Count the occurrences of each label in the training set
    label_counts = Counter(train_labels)

    # Calculate the total number of samples in the training set
    total_samples = len(train_labels)

    # Print the percentage of the training set for each label
    for label, count in label_counts.items():
        percentage = (count / total_samples) * 100
        print(f"Label '{label}': {percentage:.2f}% of the training set, Exact count: {count}")

    # print the total number of different labels in the training set:
    print(f"Total number of different labels in the training set: {len(label_counts)}")

    #adsValData = DataLoader(ds_val, batch_size=2, shuffle=False)
    # print adsValData type
    #print('adsValData type: ', type(adsValData))

    train_size = len(ds_train)
    val_size = len(ds_val)
    print('train_size: ',train_size)
    print('val_size: ',val_size)

    cal_max_epochs = cal_max_epochs(max_epochs, cal_weightage(train_size))
    print("max epochs set to: ", cal_max_epochs)
    dm = DataModule(
        ds_train = ds_train,
        ds_val = ds_val,
        #ds_test = ds_test,
        batch_size=1,
        num_workers=16,
        pin_memory=True,
    )
    model_name = os.getenv('MODEL_NAME', 'ResNet101')

    # Initialize the model
    model = select_model(model_name)
    print(f"Using model: {model_name}")
    to_monitor = "val/AUC_ROC"
    min_max = "max"
    log_every_n_steps = 1
    early_stopping = EarlyStopping(
        monitor=to_monitor,
        min_delta=0.0,  # minimum change in the monitored quantity to qualify as an improvement
        patience=10,  # number of checks with no improvement
        mode=min_max
    )
    checkpointing = ModelCheckpoint(
        dirpath=str(path_run_dir),  # dirpath
        monitor=to_monitor,
        #every_n_train_steps=log_every_n_steps,
        save_last=True,
        save_top_k=2,
        #filename='odelia-epoch{epoch:02d}-val_AUC_ROC{val/AUC_ROC:.2f}',
        mode=min_max,
    )
    useCuda = torch.cuda.is_available()


    lFArgsDict={}
    lFArgsDict['reduction']='sum'
    mFArgsDict={}
    mFArgsDict['task']="multiclass"
    mFArgsDict['num_classes']=2

    #model = model.to(torch.device('cuda'))
    if local_compare_flag:
        torch.autograd.set_detect_anomaly(True)
        trainer = Trainer(
            accelerator='gpu', devices=1,
            precision=16,
            default_root_dir=str(path_run_dir),
            callbacks=[checkpointing, early_stopping],
            enable_checkpointing=True,
            check_val_every_n_epoch=1,
            min_epochs=50,
            log_every_n_steps=log_every_n_steps,
            auto_lr_find=False,
            max_epochs=120,
            num_sanity_val_steps=2,
            logger=TensorBoardLogger(save_dir=path_run_dir)
        )
        trainer.fit(model, datamodule=dm)
    else:
        #TODO: enable sl loss calculation
        swarmCallback = SwarmCallback(
                                      totalEpochs=cal_max_epochs,
                                      syncFrequency=1024,
                                      minPeers=min_peers,
                                      maxPeers=max_peers,
                                      #adsValData=adsValData,
                                      #adsValBatchSize=2,
                                      nodeWeightage=cal_weightage(train_size),
                                      model=model,
                                      #lossFunction="BCEWithLogitsLoss",
                                      #lossFunctionArgs=lFArgsDict,
                                      #metricFunction="F1Score",
                                      #metricFunctionArgs=mFArgsDict
        )

        torch.autograd.set_detect_anomaly(True)
        swarmCallback.logger.setLevel(logging.DEBUG)
        swarmCallback.on_train_begin()
        trainer = Trainer(
            accelerator='gpu', devices=1,
            precision=16,
            default_root_dir=str(path_run_dir),
            callbacks=[checkpointing, User_swarm_callback(swarmCallback)],#early_stopping
            enable_checkpointing=True,
            check_val_every_n_epoch=1,
            #min_epochs=5,
            log_every_n_steps=log_every_n_steps,
            auto_lr_find=False,
            max_epochs=cal_max_epochs,
            num_sanity_val_steps=2,
            logger=TensorBoardLogger(save_dir=path_run_dir)
        )
        trainer.fit(model, datamodule=dm)
        swarmCallback.on_train_end()
    model.save_best_checkpoint(trainer.logger.log_dir, checkpointing.best_model_path)
    model.save_last_checkpoint(trainer.logger.log_dir, checkpointing.last_model_path)

    import subprocess

    # Get the container ID for the latest user-env container
    get_container_id_command = 'docker ps -a --filter "name=us*" --format "{{.ID}}" | head -n 1'
    container_id = subprocess.check_output(get_container_id_command, shell=True, text=True).strip()

    # Get the latest log for the user-env container
    get_logs_command = f"docker logs {container_id}"
    logs_process = subprocess.Popen(get_logs_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Print and log the output
    with open(os.path.join(path_run_dir,"container_logs.txt"), "w") as log_file:
        for line in logs_process.stdout:
            line = line.decode("utf-8").rstrip()
            print(line)
            log_file.write(line + "\n")
    predict(path_run_dir, os.path.join(dataDir, task_data_name,'test'), model_name)
    predict_last(path_run_dir, os.path.join(dataDir, task_data_name,'test'), model_name)

