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
from models import ResNet
from predict import predict

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
    # ------------ Settings/Defaults ----------------
    task_data_name = 'WP1'
    scratchDir = os.getenv('SCRATCH_DIR', '/platform/scratch')
    dataDir = os.getenv('DATA_DIR', '/platform/data/')
    max_epochs = int(os.getenv('MAX_EPOCHS', 100))
    min_peers = int(os.getenv('MIN_PEERS', 2))
    max_peers = int(os.getenv('MAX_PEERS', 7))
    local_compare_flag = bool(os.getenv('LOCAL_COMPARE_FLAG', False))
    useAdaptiveSync = bool(os.getenv('USE_ADAPTIVE_SYNC', False))
    syncFrequency = int(os.getenv('SYNC_FREQUENCY', 512))

    #print(os.getenv('DATA_DIR'))
    #print(f"Using {scratchDir} for training")
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    path_run_dir = os.path.join(scratchDir, (str(current_time)+ '_' +task_data_name + 'swarm_learning'))  # !
    # path_run_dir = Path.cwd() / 'runs' / str(current_time)
    #path_run_dir.mkdir(parents=True, exist_ok=True)
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    print(f"Using {accelerator} for training")

    # ------------ Load Data ----------------
    #dataDir = os.getenv('DATA_DIR', '/tmp/test/host1-partial-data')  # !
    #print current directory
    print("Current Directory " , os.getcwd())
    ds = DUKE_Dataset3D(
        flip=True,
        path_root=os.path.join(dataDir, task_data_name,'train_val')
    )
    train_size = int(0.8 * len(ds))
    val_size = int(0.2 * len(ds))
    ds_train = Subset(ds, list(range(train_size)))
    ds_val = Subset(ds, list(range(train_size, train_size+val_size)))
    #ds_test = Subset(ds, list(range(train_size + val_size, len(ds))))
    print('train_size: ',train_size)
    print('val_size: ',val_size)
    #print('test_size: ',ds_train)
    #print(ds_train[0])
    #print(ds_val[0])
    #print(ds_test[0])

    dm = DataModule(
        ds_train = ds_train,
        ds_val = ds_val,
        #ds_test = ds_test,
        batch_size=1,
        # num_workers=0,
        pin_memory=True,
    )

    #print('========1========')

    # ------------ Initialize Model ------------
    model = ResNet(in_ch=1, out_ch=1, spatial_dims=3)


    # -------------- Training Initialization ---------------
    to_monitor = "val/AUC_ROC"
    min_max = "max"
    log_every_n_steps = 1
    #print('========2========')
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
    swarmCallback = SwarmCallback(syncFrequency=syncFrequency,
                                  minPeers=min_peers,
                                  maxPeers=max_peers,
                                  adsValData=ds_val,
                                  adsValBatchSize=2,
                                  nodeWeightage=cal_weightage(train_size),
                                  model=model)
    torch.autograd.set_detect_anomaly(True)
    #print('========3========')
    swarmCallback.logger.setLevel(logging.DEBUG)
    swarmCallback.on_train_begin()  # !
    #print('========4========')
    #for epoch in range(max_expochs):
        #print('---------epoch: ', epoch, '---------')
    trainer = Trainer(
        accelerator=accelerator,
        # devices=[0],
        precision=16,
        # gradient_clip_val=0.5,
        default_root_dir=str(path_run_dir),
        callbacks=[checkpointing, User_swarm_callback(swarmCallback)],#early_stopping
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        min_epochs=5,
        log_every_n_steps=log_every_n_steps,
        auto_lr_find=False,
        # limit_val_batches=0, # 0 = disable validation - Note: Early Stopping no longer available
        max_epochs=max_epochs,
        num_sanity_val_steps=2,
        logger=TensorBoardLogger(save_dir=path_run_dir)
    )
    trainer.fit(model, datamodule=dm)
        #swarmCallback.on_epoch_end()
    # ---------------- Execute Training ----------------
    swarmCallback.on_train_end()  # !
    #print('========5========')

    # ------------- Save path to best model -------------
    model.save_best_checkpoint(trainer.logger.log_dir, checkpointing.best_model_path)
    #print('========6========')

    predict(path_run_dir, os.path.join(dataDir, task_data_name,'test'))