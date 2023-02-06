import logging
from pathlib import Path
from datetime import datetime

import torch
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data.dataset import Subset

from data.datasets import DUKE_Dataset3D
from data.datamodules import DataModule

import os  # !
import logging

from models import BasicClassifier
import monai.networks.nets as nets
import torch
import torch.nn.functional as F
from swarmlearning.pyt import SwarmCallback

# ------------ Swarm Callback ------------ #!



class ResNet(BasicClassifier):
    def __init__(
            self,
            in_ch,
            out_ch,
            spatial_dims=3,
            block='basic',
            layers=[3, 4, 6, 3],
            block_inplanes=[64, 128, 256, 512],
            feed_forward=True,
            loss=torch.nn.BCEWithLogitsLoss,
            loss_kwargs={},
            optimizer=torch.optim.AdamW,
            optimizer_kwargs={'lr': 1e-4},
            lr_scheduler=None,
            lr_scheduler_kwargs={},
            aucroc_kwargs={"task": "binary"},
            acc_kwargs={"task": "binary"}
    ):
        super().__init__(in_ch, out_ch, spatial_dims, loss, loss_kwargs, optimizer, optimizer_kwargs, lr_scheduler,
                         lr_scheduler_kwargs)
        self.model = nets.ResNet(block, layers, block_inplanes, spatial_dims, in_ch, 7, 1, False, 'B', 1.0, out_ch,
                                 feed_forward, True)

    def forward(self, x_in, **kwargs):
        pred_hor = self.model(x_in)
        return pred_hor
'''
    def training_step(self, batch, batch_idx):
        x, y = batch['source'], batch['target']
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        swarmCallback.on_batch_end()
        return loss
'''
from pytorch_lightning.callbacks import Callback


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

max_expochs = 100
if __name__ == "__main__":
    # ------------ Settings/Defaults ----------------
    scratchDir = os.getenv('SCRATCH_DIR', '/platform/scratch')  # !
    dataDir = os.getenv('DATA_DIR', '/platform/data')
    print(os.getenv('DATA_DIR'))
    print(f"Using {scratchDir} for training")
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    path_run_dir = os.path.join(scratchDir, str(current_time))  # !
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
        #path_root="/tmp/test"
        path_root=dataDir
        # path_root = '/mnt/sda1/swarm-learning/radiology-dataset/odelia_dataset_unilateral_256x256x32/'
    )
    print("++++++++++")
    print('len(ds): ',len(ds))
    #print(ds[0])

    # WARNING: Very simple split approach
    train_size = int(0.64 * len(ds))
    val_size = int(0.16 * len(ds))
    test_size = len(ds) - train_size - val_size
    ds_train = Subset(ds, list(range(train_size)))
    ds_val = Subset(ds, list(range(train_size, train_size + val_size)))
    ds_test = Subset(ds, list(range(train_size + val_size, len(ds))))
    print('train_size: ',train_size)
    print('val_size: ',val_size)
    print('test_size: ',ds_train)
    #print(ds_train[0])
    #print(ds_val[0])
    #print(ds_test[0])


    dm = DataModule(
        ds_train = ds_train,
        ds_val = ds_val,
        ds_test = ds_test,
        batch_size=1,
        # num_workers=0,
        pin_memory=True,
    )

    print('========1========')

    # ------------ Initialize Model ------------
    model = ResNet(in_ch=1, out_ch=1, spatial_dims=3)


    # -------------- Training Initialization ---------------
    to_monitor = "val/AUC_ROC"
    min_max = "max"
    log_every_n_steps = 1
    print('========2========')
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
        save_top_k=10,
        filename='sample-mnist-epoch{epoch:02d}-val_AUC_ROC{val/AUC_ROC:.2f}',
        mode=min_max,
    )
    useCuda = torch.cuda.is_available()

    device = torch.device("cuda" if useCuda else "cpu")
    model = model.to(torch.device(device))
    swarmCallback = SwarmCallback(syncFrequency=1024,
                                  minPeers=2,
                                  useAdaptiveSync=False,
                                  adsValData=ds_val,
                                  adsValBatchSize=2,
                                  model=model)
    torch.autograd.set_detect_anomaly(True)
    print('========3========')
    swarmCallback.logger.setLevel(logging.DEBUG)
    swarmCallback.on_train_begin()  # !
    print('========4========')
    #for epoch in range(max_expochs):
        #print('---------epoch: ', epoch, '---------')
    trainer = Trainer(
        accelerator=accelerator,
        # devices=[0],
        precision=16,
        # gradient_clip_val=0.5,
        default_root_dir=str(path_run_dir),
        callbacks=[checkpointing, early_stopping, User_swarm_callback(swarmCallback)],
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        min_epochs=20,
        log_every_n_steps=log_every_n_steps,
        auto_lr_find=False,
        # limit_val_batches=0, # 0 = disable validation - Note: Early Stopping no longer available
        max_epochs=50,
        num_sanity_val_steps=2,
        logger=TensorBoardLogger(save_dir=path_run_dir)
    )
    trainer.fit(model, datamodule=dm)
        #swarmCallback.on_epoch_end()
    # ---------------- Execute Training ----------------
    swarmCallback.on_train_end()  # !
    print('========5========')

    # ------------- Save path to best model -------------
    model.save_best_checkpoint(trainer.logger.log_dir, checkpointing.best_model_path)
    print('========6========')

# ? where place "swarmCallback.on_batch_end()" and "swarmCallback.on_epoch_end(epoch)"

