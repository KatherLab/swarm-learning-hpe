import glob2
import torchio as tio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision.io import read_image
from scipy import signal
import random
import math
import os
import torch

PREPROCESSING_TRANSORMS = tio.Compose([
    tio.RescaleIntensity(out_min_max=(0, 1)),
    tio.CropOrPad(target_shape=( 1, 1024, 1024)),
])


TRAIN_TRANSFORMS = tio.Compose([
    tio.RandomAffine( degrees=(-5,5,0,0,0,0), scales = 0, default_pad_value = 'minimum',p =0.5),
    tio.RandomFlip(axes=(2), flip_probability=0.5),
    #tio.RandomGamma(log_gamma=(-0.3, 0.3))
])


VAL_TRANSFORMS = None


class NIHXRayDatasetSubwise(Dataset):
    def __init__(self, root_dir,  split='train', training_samples = 1000, validation_samples = 200, augmentation = False, downsample = 2):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.training_samples = training_samples
        self.validation_samples = validation_samples
        self.preprocessing = PREPROCESSING_TRANSORMS
        self.transforms = TRAIN_TRANSFORMS 
        self.downsample_transform = tio.Resize((1, 1024//downsample, 1024//downsample))
        self.augmentation = augmentation
        self.downsample = downsample
        self.paths = self._get_file_paths()
        self.labels, self.idxs = self._get_labels()


    def _get_file_paths(self):
        file_paths = glob2.glob(self.root_dir + '/*/*/*.png*')
        file_paths = [i.rsplit('_',1)[0] for i in file_paths]
        file_paths = list(set(file_paths))
        file_paths.sort()
        file_paths = [i+'_000.png' for i in file_paths if os.path.isfile(i+'_000.png')]
        file_paths = file_paths[0:self.training_samples] if (self.split == 'train') else file_paths[-self.validation_samples:] 
        return file_paths

    def _get_labels(self):
        labels_df = pd.read_csv(self.root_dir + '/Data_Entry_2017.csv')
        labels_df = labels_df[labels_df['Image Index'].str.contains("000.png")]
        labels = labels_df['Finding Labels'].str.get_dummies(sep='|')
        labels = labels.iloc[0:self.training_samples] if (self.split == 'train') else labels.iloc[ labels.index[-self.validation_samples:]]
        idxs = labels_df['Image Index']
        idxs = idxs.iloc[0:self.training_samples] if (self.split == 'train') else idxs.iloc[ idxs.index[-self.validation_samples:]]
 
        return labels, idxs
        
    def _get_sampler_weights(self):
        labels_np = self.labels.to_numpy()
        labels_np = np.argmax(labels_np, axis=1)
        class_sample_count = np.array(
            [len(np.where(labels_np == t)[0]) for t in np.arange(self.labels.shape[-1])])
        weight = 1. / class_sample_count
        samples_weight = torch.from_numpy(np.array([weight[t] for t in labels_np]))    
        return samples_weight

    def __len__(self):
        return len(self.paths)


    def __getitem__(self, index: int):
        img = read_image(self.paths[index]).unsqueeze(dim=1)
        img = self.preprocessing(img)
        if self.downsample>1:
            img = self.downsample_transform(img)      
        if self.augmentation:
            img = self.transforms(img)         
        label = torch.tensor(self.labels.iloc[index,:])    
        img_id =  self.idxs.iloc[index]    
        return {'data': img[[0],0,:], 'cond': label, 'path': self.paths[index], 'img_id': img_id, 'index':index }
