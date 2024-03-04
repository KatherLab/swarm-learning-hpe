import glob2
import torchio as tio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision.io import read_image
from scipy import signal
import random
from pydicom import dcmread
import argparse
import torch
import os
PREPROCESSING_TRANSORMS = tio.Compose([
    tio.RescaleIntensity(out_min_max=(0, 1)),
    tio.CropOrPad(target_shape=( 1, 320, 320)),
])


TRAIN_TRANSFORMS = tio.Compose([
    tio.RandomAffine( degrees=(-5,5,0,0,0,0), scales = 0, default_pad_value = 'minimum',p =0.5),
    tio.RandomFlip(axes=(2), flip_probability=0.5),
    #tio.RandomGamma(log_gamma=(-0.3, 0.3))
])


VAL_TRANSFORMS = None


class fastMRIDataset(Dataset):
    def __init__(self, root_dir,  split='train', training_samples = 1000, validation_samples = 200, augmentation = False, donwsample = 1):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.training_samples = training_samples
        self.validation_samples = validation_samples
        self.preprocessing = PREPROCESSING_TRANSORMS
        self.transforms = TRAIN_TRANSFORMS 
        self.donwsample_transform = tio.Resize((1, 320//donwsample, 320//donwsample))
        self.augmentation = augmentation
        self.downsample = donwsample
        self.paths = self._get_file_paths()

    def resizing(self, img ,  size = 320):
        ratio = img.shape[-2] / size
        resizing_transform = tio.Resize((1, size, int(img.shape[-1]/ratio)))
        img = resizing_transform(img)
        return img


    def _get_file_paths(self):

        file_paths = glob2.glob(self.root_dir + '/*/*.dcm*')
        file_paths.sort()
        if os.path.isfile(self.root_dir+ '/centered_indexes.npz'):
            idx = np.load(self.root_dir+ '/centered_indexes.npz')['idx']
        else:
            aa = []
            for ii in range(len(file_paths)):
                img = dcmread(file_paths[ii])
                img = img.pixel_array/img.pixel_array.max()
                #check where 75 percentile of normalized slice is
                aa.append(np.percentile(img.flatten(),75))    
            aa = np.asanyarray(aa)
            #indeces for which 75 percentile is less than 0.1
            idx_prime = np.where(aa<0.1)[0]
            idx = np.setdiff1d(range(len(file_paths)),idx_prime)
        #select only centered slices    
        file_paths = [file_paths[index] for index in idx]
        file_paths = file_paths[0:self.training_samples] if (self.split == 'train') else file_paths[-self.validation_samples:] 
        return file_paths


    def __len__(self):
        return len(self.paths)

    def _get_sampler_weights(self):
        labels = [];file_name = []; seq =[]
        for idx in range(len(self.paths)):  
            dcm = dcmread(self.paths[idx])
            contrast = dcm[0x0008, 0x103e].value
            #print(contrast)
            if 'T1' in contrast and 'POST' in contrast: labels.append(0)
            elif 'T1' in contrast and not('POST' in contrast): labels.append(1)
            elif 'T2' in contrast and not('FLAIR' in contrast) : labels.append(2) 
            elif 'FLAIR' in contrast :labels.append(3)  

            seq.append(contrast)
            file_name.append(self.paths[idx])        
        class_sample_count = np.array(
            [len(np.where(labels == t)[0]) for t in np.arange(np.unique(labels).shape[-1])])
        weight = 1. / class_sample_count
        samples_weight = torch.from_numpy(np.array([weight[t] for t in labels]))  
        return samples_weight

    def __getitem__(self, index: int):
        dcm = dcmread(self.paths[index])        
        #print(dcm.pixel_array.dtype)
        img= torch.tensor(np.array(dcm.pixel_array, dtype = 'int32'))
        img = self.resizing(img.unsqueeze(dim=0).unsqueeze(dim=0))
        img = self.preprocessing(img)
        contrast = dcm[0x0008, 0x103e].value
        if 'T1' in contrast and 'POST' in contrast: label = torch.tensor([0])  
        elif 'T1' in contrast and not('POST' in contrast): label = torch.tensor([1])  
        elif 'T2' in contrast and not('FLAIR' in contrast) : label = torch.tensor([2])  
        elif 'FLAIR' in contrast :label = torch.tensor([3])  
        if self.downsample>1:
            img = self.donwsample_transform(img)      
        if self.augmentation:
            img = self.transforms(img)         
        return {'data': img[[0],0,:], 'cond': label, 'path': self.paths[index], 'contrast' :contrast}
