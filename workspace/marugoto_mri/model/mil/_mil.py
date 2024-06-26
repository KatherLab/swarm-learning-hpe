from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, TypeVar
from pathlib import Path
import os
from .transformer import Transformer
from .ViT import ViT
from .timmViT import MILModel
import numpy as np

import logging
from swarmlearning.pyt import SwarmCallback

from fastai.callback.core import *
from fastai.vision.all import (
    Learner,
    DataLoader,
    DataLoaders,
    RocAuc,
    SaveModelCallback,
    CSVLogger,
)
import h5py
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import numpy.typing as npt
import pandas as pd
import sys
sys.path.append("..")
from common.data import SKLearnEncoder

from .data import make_dataset
#from .model import MILModel


__all__ = ["train", "deploy"]


T = TypeVar("T")
class User_swarm_callback(TrainEvalCallback):
    def __init__(self, swarmCallback, **kwargs):
        super().__init__(**kwargs)
        self.swarmCallback = swarmCallback

    #def on_train_start(self, trainer, pl_module):
    #    self.swarmCallback.on_train_begin()

    def after_batch(self):
        self.swarmCallback.on_batch_end()

    def after_epoch(self):
        self.swarmCallback.on_epoch_end()

    #def on_train_end(self, trainer, pl_module):
    #    self.swarmCallback.on_train_end()

def train(
    *,
    bags: Sequence[Iterable[Path]],
    targets: Tuple[SKLearnEncoder, npt.NDArray],
    add_features: Iterable[Tuple[SKLearnEncoder, npt.NDArray]] = [],
    valid_idxs: npt.NDArray[np.int_],
    n_epoch: int = 50,
    path: Optional[Path] = None,
    local_compare_flag = False,
    min_peers = 1,
    max_peers = 5,
    syncFrequency = 32,
    useAdaptiveSync = False,
    model_type: str = "transformer",
) -> Learner:
    """Train a MLP on image features.

    Args:
        bags:  H5s containing bags of tiles.
        targets:  An (encoder, targets) pair.
        add_features:  An (encoder, targets) pair for each additional input.
        valid_idxs:  Indices of the datasets to use for validation.
    """
    target_enc, targs = targets
    batch_size = 8
    train_ds = make_dataset(
        bags=bags[~valid_idxs],  # type: ignore  # arrays cannot be used a slices yet
        targets=(target_enc, targs[~valid_idxs]),
        add_features=[(enc, vals[~valid_idxs]) for enc, vals in add_features],
        bag_size=32,
    )


    valid_ds = make_dataset(
        bags=bags[valid_idxs],  # type: ignore  # arrays cannot be used a slices yet
        targets=(target_enc, targs[valid_idxs]),
        add_features=[(enc, vals[valid_idxs]) for enc, vals in add_features],
        bag_size=None,
    )
    print('training and validation dataset size: ', len(train_ds), len(valid_ds))
    # build dataloaders
    train_dl = DataLoader(
        train_ds, batch_size=32, shuffle=True, num_workers=1, drop_last=True
    )
    valid_dl = DataLoader(
        valid_ds, batch_size=1, shuffle=False, num_workers=os.cpu_count()
    )
    batch = train_dl.one_batch()

    # TODO: use GPU for training
    #useCuda = torch.cuda.is_available()
    #device = torch.device("cuda" if useCuda else "cpu")
    if model_type == "transformer":
        model = ViT(num_classes=2)  # Transformer(num_classes=2)
        #model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))  #
    elif model_type == "attmil":
        model = MILModel(batch[0].shape[-1], batch[-1].shape[-1])
    elif model_type == "timmViT":
        model = MILModel(n_classes=2)
    else:
        raise ValueError(f"Unknown model type {model_type}")
    #model = model.to(torch.device(device))




    # weigh inversely to class occurances
    counts = pd.value_counts(targs[~valid_idxs])
    weight = counts.sum() / counts
    weight /= weight.sum()
    # reorder according to vocab
    weight = torch.tensor(
        list(map(weight.get, target_enc.categories_[0])), dtype=torch.float32
    )
    loss_func = nn.CrossEntropyLoss(weight=weight)

    dls = DataLoaders(train_dl, valid_dl)

    if local_compare_flag:
        print('local compare flag is set')
        learn = Learner(dls, model, loss_func=loss_func, metrics=[RocAuc()], path=path)
        cbs = [
            SaveModelCallback(fname=f"best_valid"),
            CSVLogger(),
        ]
        learn.fit_one_cycle(n_epoch=n_epoch, lr_max=1e-4, cbs=cbs)
        return learn, None
    else:
        swarmCallback = SwarmCallback(syncFrequency=syncFrequency,
                                      minPeers=min_peers,
                                      maxPeers=max_peers,
                                      adsValData=valid_ds,
                                      adsValBatchSize=2,
                                      nodeWeightage=cal_weightage(len(train_ds)),
                                      model=model)
        swarmCallback.logger.setLevel(logging.DEBUG)
        print('local compare flag is not set')
        swarmCallback.on_train_begin()
        learn = Learner(dls, model, loss_func=loss_func, metrics=[RocAuc()], path=path)
        cbs = [
            SaveModelCallback(fname=f"best_valid"),
            CSVLogger(),
            User_swarm_callback(swarmCallback),
        ]

        learn.fit_one_cycle(n_epoch=30, lr_max=1e-4, cbs=cbs)


        return learn, swarmCallback

def cal_weightage(train_size):
    full_dataset_size = 922
    return int(100 * train_size / full_dataset_size)
def deploy(
    test_df: pd.DataFrame,
    learn: Learner,
    *,
    target_label: Optional[str] = None,
    cat_labels: Optional[Sequence[str]] = None,
    cont_labels: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    assert test_df.PATIENT.nunique() == len(test_df), "duplicate patients!"
    if target_label is None:
        target_label = learn.target_label
    if cat_labels is None:
        cat_labels = learn.cat_labels
    if cont_labels is None:
        cont_labels = learn.cont_labels

    target_enc = learn.dls.dataset._datasets[-1].encode
    categories = target_enc.categories_[0]
    add_features = []
    if cat_labels:
        cat_enc = learn.dls.dataset._datasets[-2]._datasets[0].encode
        add_features.append((cat_enc, test_df[cat_labels].values))
    if cont_labels:
        cont_enc = learn.dls.dataset._datasets[-2]._datasets[1].encode
        add_features.append((cont_enc, test_df[cont_labels].values))

    test_ds = make_dataset(
        bags=test_df.slide_path.values,
        targets=(target_enc, test_df[target_label].values),
        add_features=add_features,
        bag_size=None,
    )

    test_dl = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=os.cpu_count()
    )

    # removed softmax in forward, but add here to get 0-1 probabilities
    patient_preds, patient_targs = learn.get_preds(dl=test_dl, act=nn.Softmax(dim=1))

    # make into DF w/ ground truth
    patient_preds_df = pd.DataFrame.from_dict(
        {
            "PATIENT": test_df.PATIENT.values,
            target_label: test_df[target_label].values,
            **{
                f"{target_label}_{cat}": patient_preds[:, i]
                for i, cat in enumerate(categories)
            },
        }
    )

    # calculate loss
    patient_preds = patient_preds_df[
        [f"{target_label}_{cat}" for cat in categories]
    ].values
    patient_targs = target_enc.transform(
        patient_preds_df[target_label].values.reshape(-1, 1)
    )
    patient_preds_df["loss"] = F.cross_entropy(
        torch.tensor(patient_preds), torch.tensor(patient_targs), reduction="none"
    )

    patient_preds_df["pred"] = categories[patient_preds.argmax(1)]

    # reorder dataframe and sort by loss (best predictions first)
    patient_preds_df = patient_preds_df[
        [
            "PATIENT",
            target_label,
            "pred",
            *(f"{target_label}_{cat}" for cat in categories),
            "loss",
        ]
    ]
    patient_preds_df = patient_preds_df.sort_values(by="loss")

    return patient_preds_df
