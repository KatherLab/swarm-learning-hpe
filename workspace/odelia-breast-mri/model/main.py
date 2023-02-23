from pathlib import Path
from datetime import datetime

from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data.dataset import Subset

from data.datasets import DUKE_Dataset3D
from data.datamodules import DataModule

import os
from pathlib import Path
import logging
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data.dataset import Subset
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from data.datasets import DUKE_Dataset3D
from data.datamodules import DataModule
from utils.roc_curve import plot_roc_curve, cm2acc, cm2x

from models import BasicClassifier
import monai.networks.nets as nets
import torch
import torch.nn.functional as F
from swarmlearning.pyt import SwarmCallback
from pytorch_lightning.callbacks import Callback
#from main_predict import predict
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

def predict(model_dir, test_data_dir):
    # ------------ Settings/Defaults ----------------
    # path_run = Path.cwd() / 'runs/2023_02_06_175325'
    # path_run = Path('/opt/hpe/swarm-learning-hpe/workspace/odelia-breast-mri/user-odelia-breast-mri-192.168.33.102/data-and-scratch/scratch/2023_02_06_205810/')
    path_run = Path(model_dir)
    #path_run = Path(
        #'/opt/hpe/swarm-learning-hpe/workspace/odelia-breast-mri/user-odelia-breast-mri-192.168.33.102/data-and-scratch/scratch/2023_02_06_224851')
    #path_out = Path().cwd() / 'results' / path_run.name
    path_out = Path(path_run) / 'results' / path_run.name
    path_out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fontdict = {'fontsize': 10, 'fontweight': 'bold'}

    # ------------ Logging --------------------
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    # ------------ Load Data ----------------
    ds = DUKE_Dataset3D(
        flip=True,
        path_root=test_data_dir
    )

    ds_test = ds

    dm = DataModule(
        ds_test=ds_test,
        batch_size=1,
        # num_workers=0,
        # pin_memory=True,
    )

    # ------------ Initialize Model ------------
    model = ResNet.load_best_checkpoint(path_run, version=0)
    model.to(device)
    model.eval()

    results = {'GT': [], 'NN': [], 'NN_pred': []}
    for batch in tqdm(dm.test_dataloader()):
        source, target = batch['source'], batch['target']

        # Run Model
        pred = model(source.to(device)).cpu()
        pred = torch.sigmoid(pred)
        pred_binary = torch.argmax(pred, dim=1)

        results['GT'].extend(target.tolist())
        results['NN'].extend(pred_binary.tolist())
        results['NN_pred'].extend(pred[:, 0].tolist())

    df = pd.DataFrame(results)
    df.to_csv(path_out / 'results.csv')

    #  -------------------------- Confusion Matrix -------------------------
    cm = confusion_matrix(df['GT'], df['NN'])
    tn, fp, fn, tp = cm.ravel()
    n = len(df)
    logger.info(
        "Confusion Matrix: TN {} ({:.2f}%), FP {} ({:.2f}%), FN {} ({:.2f}%), TP {} ({:.2f}%)".format(tn, tn / n * 100,
                                                                                                      fp, fp / n * 100,
                                                                                                      fn, fn / n * 100,
                                                                                                      tp, tp / n * 100))

    # ------------------------------- ROC-AUC ---------------------------------
    fig, axis = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    y_pred_lab = np.asarray(df['NN_pred'])
    y_true_lab = np.asarray(df['GT'])
    tprs, fprs, auc_val, thrs, opt_idx, cm = plot_roc_curve(y_true_lab, y_pred_lab, axis, fontdict=fontdict)
    fig.tight_layout()
    fig.savefig(path_out / f'roc.png', dpi=300)

    #  -------------------------- Confusion Matrix -------------------------
    acc = cm2acc(cm)
    _, _, sens, spec = cm2x(cm)
    df_cm = pd.DataFrame(data=cm, columns=['False', 'True'], index=['False', 'True'])
    fig, axis = plt.subplots(1, 1, figsize=(4, 4))
    sns.heatmap(df_cm, ax=axis, cbar=False, fmt='d', annot=True)
    axis.set_title(f'Confusion Matrix ACC={acc:.2f}', fontdict=fontdict)  # CM =  [[TN, FP], [FN, TP]]
    axis.set_xlabel('Prediction', fontdict=fontdict)
    axis.set_ylabel('True', fontdict=fontdict)
    fig.tight_layout()
    fig.savefig(path_out / f'confusion_matrix.png', dpi=300)

    logger.info(f"Malign  Objects: {np.sum(y_true_lab)}")
    logger.info("Confusion Matrix {}".format(cm))
    logger.info("Sensitivity {:.2f}".format(sens))
    logger.info("Specificity {:.2f}".format(spec))

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
    task_data_name = '40-30-10-20'
    scratchDir = os.getenv('SCRATCH_DIR', '/platform/scratch')
    dataDir = os.getenv('DATA_DIR', '/platform/data/')
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
        save_top_k=2,
        #filename='odelia-epoch{epoch:02d}-val_AUC_ROC{val/AUC_ROC:.2f}',
        mode=min_max,
    )
    useCuda = torch.cuda.is_available()

    device = torch.device("cuda" if useCuda else "cpu")
    model = model.to(torch.device(device))
    swarmCallback = SwarmCallback(syncFrequency=512,
                                  minPeers=3,
                                  useAdaptiveSync=False,
                                  adsValData=ds_val,
                                  adsValBatchSize=2,
                                  nodeWeightage=100,
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
        min_epochs=50,
        log_every_n_steps=log_every_n_steps,
        auto_lr_find=False,
        # limit_val_batches=0, # 0 = disable validation - Note: Early Stopping no longer available
        max_epochs=max_expochs,
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