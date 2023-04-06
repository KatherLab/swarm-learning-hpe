from datetime import datetime
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import os
from pathlib import Path
import logging
from tqdm import tqdm
import numpy as np
from torch.utils.data.dataset import Subset
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from data.datasets import DUKE_Dataset3D
from data.datamodules import DataModule
from utils.roc_curve import plot_roc_curve, cm2acc, cm2x
import monai.networks.nets as nets
import torch
from swarmlearning.pyt import SwarmCallback
from pytorch_lightning.callbacks import Callback
from models import ResNet, VisionTransformer, EfficientNet, EfficientNet3D, EfficientNet3Db7, DenseNet121, UNet3D
from predict import predict
from predict_last import predict_last

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

def cal_weightage(train_size):
    full_dataset_size = 922
    return int(100 * train_size / full_dataset_size)

if __name__ == "__main__":
    task_data_name = 'DUKE'
    scratchDir = os.getenv('SCRATCH_DIR', '/platform/scratch')
    dataDir = os.getenv('DATA_DIR', '/platform/data/')
    max_epochs = int(os.getenv('MAX_EPOCHS', 100))
    min_peers = int(os.getenv('MIN_PEERS', 2))
    max_peers = int(os.getenv('MAX_PEERS', 7))
    local_compare_flag = os.getenv('LOCAL_COMPARE_FLAG', 'False').lower() == 'true'
    useAdaptiveSync = os.getenv('USE_ADAPTIVE_SYNC', 'False').lower() == 'true'
    syncFrequency = int(os.getenv('SYNC_FREQUENCY', 512))
    model_name = os.getenv('MODEL_NAME', 'ResNet50')
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    if local_compare_flag:
        print("Running in local compare mode")
        path_run_dir = os.path.join(scratchDir, (str(current_time)+ '_' +task_data_name + '_' + model_name + '_local_compare'))
    else:
        path_run_dir = os.path.join(scratchDir, (str(current_time)+ '_' +task_data_name + '_' + model_name + '_swarm_learning'))

    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    print(f"Using {accelerator} for training")

    print("Current Directory " , os.getcwd())
    ds = DUKE_Dataset3D(
        flip=True,
        path_root=os.path.join(dataDir, task_data_name,'train_val')
    )
    train_size = int(0.8 * len(ds))
    val_size = int(0.2 * len(ds))
    ds_train = Subset(ds, list(range(train_size)))
    ds_val = Subset(ds, list(range(train_size, train_size+val_size)))
    print('train_size: ',train_size)
    print('val_size: ',val_size)


    dm = DataModule(
        ds_train = ds_train,
        ds_val = ds_val,
        #ds_test = ds_test,
        batch_size=1,
        # num_workers=0,
        pin_memory=True,
    )
    if model_name == 'ResNet18':
        layers = [2, 2, 2, 2]
    elif model_name == 'ResNet34':
        layers = [3, 4, 6, 3]
    elif model_name == 'ResNet50':
        layers = [3, 4, 6, 3]
    elif model_name == 'ResNet101':
        layers = [3, 4, 23, 3]
    elif model_name == 'ResNet152':
        layers = [3, 8, 36, 3]
    else:
        layers = None

    if layers is not None:
        # ------------ Initialize Model ------------
        model = ResNet(in_ch=1, out_ch=1, spatial_dims=3, layers=layers)
    elif model_name in ['efficientnet_l1', 'efficientnet_l2', 'efficientnet_b4', 'efficientnet_b7']:
        model = EfficientNet(model_name=model_name, in_ch=1, out_ch=1, spatial_dims=3)
    elif model_name == 'EfficientNet3Db0':
        blocks_args_str = [
            "r1_k3_s11_e1_i32_o16_se0.25",
            "r2_k3_s22_e6_i16_o24_se0.25",
            "r2_k5_s22_e6_i24_o40_se0.25",
            "r3_k3_s22_e6_i40_o80_se0.25",
            "r3_k5_s11_e6_i80_o112_se0.25",
            "r4_k5_s22_e6_i112_o192_se0.25",
            "r1_k3_s11_e6_i192_o320_se0.25"]
    elif model_name == 'EfficientNet3Db4':
        blocks_args_str = [
            "r1_k3_s11_e1_i48_o24_se0.25",
            "r3_k3_s22_e6_i24_o32_se0.25",
            "r3_k5_s22_e6_i32_o56_se0.25",
            "r4_k3_s22_e6_i56_o112_se0.25",
            "r4_k5_s11_e6_i112_o160_se0.25",
            "r5_k5_s22_e6_i160_o272_se0.25",
            "r2_k3_s11_e6_i272_o448_se0.25"]
    elif model_name == 'EfficientNet3Db7':
        blocks_args_str = [
            "r1_k3_s11_e1_i32_o32_se0.25",
            "r4_k3_s22_e6_i32_o48_se0.25",
            "r4_k5_s22_e6_i48_o80_se0.25",
            "r4_k3_s22_e6_i80_o160_se0.25",
            "r6_k5_s11_e6_i160_o256_se0.25",
            "r6_k5_s22_e6_i256_o384_se0.25",
            "r3_k3_s11_e6_i384_o640_se0.25"]
    elif model_name == 'DenseNet121':
        model = DenseNet121(in_ch=1, out_ch=1, spatial_dims=3)
    elif model_name == 'UNet3D':
        model = UNet3D(in_ch=1, out_ch=1, spatial_dims=3)
    else:
        raise Exception("Invalid network model specified")

    if model_name.startswith('EfficientNet3D'):
        model = EfficientNet3D(in_ch=1, out_ch=1, spatial_dims=3, blocks_args_str=blocks_args_str)
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
        #save_top_k=2,
        #filename='odelia-epoch{epoch:02d}-val_AUC_ROC{val/AUC_ROC:.2f}',
        mode=min_max,
    )
    useCuda = torch.cuda.is_available()

    device = torch.device("cuda" if useCuda else "cpu")
    model = model.to(torch.device(device))
    if local_compare_flag:
        torch.autograd.set_detect_anomaly(True)
        trainer = Trainer(
            accelerator=accelerator,
            precision=16,
            default_root_dir=str(path_run_dir),
            callbacks=[checkpointing],  # early_stopping
            enable_checkpointing=True,
            check_val_every_n_epoch=1,
            min_epochs=5,
            log_every_n_steps=log_every_n_steps,
            auto_lr_find=False,
            max_epochs=max_epochs,
            num_sanity_val_steps=2,
            logger=TensorBoardLogger(save_dir=path_run_dir)
        )
        trainer.fit(model, datamodule=dm)

        trainer.fit(model, datamodule=dm)
    else:
        swarmCallback = SwarmCallback(syncFrequency=syncFrequency,
                                      minPeers=min_peers,
                                      maxPeers=max_peers,
                                      adsValData=ds_val,
                                      adsValBatchSize=2,
                                      nodeWeightage=cal_weightage(train_size),
                                      model=model)
        torch.autograd.set_detect_anomaly(True)
        swarmCallback.logger.setLevel(logging.DEBUG)
        swarmCallback.on_train_begin()

        trainer = Trainer(
            accelerator=accelerator,
            precision=16,
            default_root_dir=str(path_run_dir),
            callbacks=[checkpointing, User_swarm_callback(swarmCallback)],#early_stopping
            enable_checkpointing=True,
            check_val_every_n_epoch=1,
            min_epochs=5,
            log_every_n_steps=log_every_n_steps,
            auto_lr_find=False,
            max_epochs=max_epochs,
            num_sanity_val_steps=2,
            logger=TensorBoardLogger(save_dir=path_run_dir)
        )
        trainer.fit(model, datamodule=dm)
        swarmCallback.on_train_end()
    model.save_best_checkpoint(trainer.logger.log_dir, checkpointing.best_model_path)
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
