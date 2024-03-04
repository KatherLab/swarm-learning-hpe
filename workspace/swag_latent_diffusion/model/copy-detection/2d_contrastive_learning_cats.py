#Load required modules
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from monai import transforms
from monai.config import print_config
from monai.data import DataLoader, Dataset
from monai.utils import first, set_determinism
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import json

import sys
sys.path.append(os.getcwd())

from generative.networks.nets.autoencoderkl import Encoder, Downsample
from pytorch_metric_learning.distances import BatchedDistance, CosineSimilarity
from pytorch_metric_learning.losses import NTXentLoss
from dataset.cats_cl import CatsDataset
from networks.SiameseNetwork import SiameseNetwork
from torch.utils.tensorboard import SummaryWriter

def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)


class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
#Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/mnt/sds/sd20i001/salman/data/Cats')
parser.add_argument('--n_epochs', type = int,default=100)
parser.add_argument('--batch_size', type = int, default=28)
parser.add_argument('--training_samples', type = int, default=1000)
parser.add_argument('--val_interval', type = int, default=1)
parser.add_argument('--base_lr', type = float, default=5e-4)#
parser.add_argument('--num_channels', type=tuple_type,  default=(64,)*7)#
parser.add_argument('--attention_levels', type=tuple_type,  default=(False, False, False))#
parser.add_argument('--save_model_interval', type = int, default=5)
parser.add_argument('--autoencoder_warm_up_n_epochs', type = int, default=10)
parser.add_argument('--ckpt_dir', type = str, default='/mnt/sds/sd20i001/salman/ckpt/Diffusion-2D-Cats/contrastive_learning_try3/')
parser.add_argument('--details', type = str, default='resent as network')
parser.add_argument('--downsample', type = int, default=2)
parser.add_argument('--loss_type', type = str, default='NTXNet')
parser.add_argument('--triplet_loss_norm', type = int, default=1)
parser.add_argument('--triplet_loss_define_loss', action='store_true')
parser.add_argument('--temperature',type = float, default=0.5)
parser.add_argument('--multi_gpu', action='store_true')
args = parser.parse_args()

#Arguments
data_dir = args.data_dir
n_epochs = args.n_epochs
batch_size =  args.batch_size
training_samples =  args.training_samples
val_interval = args.val_interval

save_model_interval = args.save_model_interval
autoencoder_warm_up_n_epochs = args.autoencoder_warm_up_n_epochs
ckpt_dir = args.ckpt_dir
num_channels = args.num_channels
downsample = args.downsample
base_lr = args.base_lr
attention_levels = (False,)*len(num_channels)
loss_type = args.loss_type
triplet_loss_norm = args.triplet_loss_norm
triplet_loss_define_loss = args.triplet_loss_define_loss
temperature = args.temperature
multi_gpu = args.multi_gpu
isExist = os.path.exists(ckpt_dir)
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(ckpt_dir)
#Save arguments
with open(ckpt_dir + '/arguments.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2) 
#Load training data
train_data = CatsDataset(root_dir= data_dir, split = 'train', training_samples = training_samples )
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
#train_data_aug = NIHXRayDataset(root_dir= data_dir, split = 'train', training_samples = training_samples , donwsample = downsample, augmentation=True)

#Load validation data
val_data = CatsDataset(root_dir= data_dir, split = 'val')
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)
#val_data_aug = NIHXRayDataset(root_dir= data_dir, split = 'val', donwsample = downsample, augmentation=True)
cosine_similarity = CosineSimilarity()

#Define autoencoder 
device = torch.device("cuda")

class Model(nn.Module):
    def __init__(self, conv_only = False, sup_training = False, num_channels = (64, 64, 64)):
        super().__init__()
        self.conv_only = conv_only  #If dense layer
        self.sup_training = sup_training  #If supervised training 
        self.attention_levels = (False,)*len(num_channels)
        self.latent_channels = 32

        self.encoder = Encoder(
            spatial_dims=2,
            in_channels=1,
            num_channels=num_channels,
            out_channels=self.latent_channels,
            num_res_blocks=(1,)*len(num_channels),
            norm_num_groups=32,
            norm_eps=1e-6,
            attention_levels=self.attention_levels,
            with_nonlocal_attn=False,
            use_flash_attention=False,
        )        
        # self.downsample = nn.Sequential(Downsample(2, self.latent_channels),
        #                         nn.SiLU(),
        #                         Downsample(2, self.latent_channels),
        #                         nn.SiLU(),
        #                         Downsample(2, self.latent_channels),
        #                         nn.SiLU(),
        #                         Downsample(2, self.latent_channels),
        #                         nn.SiLU(),
        #                         Downsample(2, self.latent_channels),
        #                         nn.SiLU())
        self.dense = nn.Sequential(nn.Flatten(),
                                nn.Linear(self.latent_channels * 4**2, 128),
                                nn.ReLU(inplace=True),
                                nn.Dropout(0.5),
                                nn.Linear(128,32))

        self.sup_layer = nn.Sequential(nn.ReLU(inplace=True),
                            nn.Linear(32,1))


    def forward(self, x):
        h = self.encoder(x)
        #h = self.downsample(h)
        if not(self.conv_only):
            h = self.dense(h)
            if self.sup_training:
                fin = self.sup_layer(h)
            else:
                fin = None
            
        return h, fin

#model = Model(num_channels = num_channels)
model = SiameseNetwork()
Siamese = model.forward_once
if multi_gpu: model = MyDataParallel(model)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

#Training
optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, eta_min = base_lr/2.0)
if loss_type =='triplet':
    if triplet_loss_define_loss:
        print('Triplt loss is cosine')
        LossTr = (
            nn.TripletMarginWithDistanceLoss(
                distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y)))    
    else:
        LossTr = torch.nn.TripletMarginLoss(p=triplet_loss_norm)
else:
    LossTr = NTXentLoss(temperature=temperature)


writer = SummaryWriter(log_dir= ckpt_dir)
epoch_losses =[]
val_losses = []
for epoch in range(n_epochs):
    
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
    progress_bar.set_description(f"Epoch {epoch}")    

    epoch_loss = 0
    model.train()
    for train_step, batch in progress_bar:
        optimizer.zero_grad()
        Pos11 = batch['data'].to(device); Pos12 = batch['data_pos'].to(device);  

        #Obtain positive and negative embddings
        #PosEmb11, _ = model(Pos11.to(device)); PosEmb12, Pred = model(Pos12.to(device))
        PosEmb11 = Siamese(Pos11.to(device)); PosEmb12 = Siamese(Pos12.to(device))
        #print(PosEmb11.max())
        
        if loss_type =='triplet':
            Neg = batch['data_neg'].to(device); NegEmb, _ = model(Neg.to(device))
            LossPos1 = LossTr(PosEmb11, PosEmb12, NegEmb )
        else:
            Labels = torch.arange(PosEmb11.shape[0])
            LossPos1 = LossTr(torch.cat((PosEmb11, PosEmb12), dim = 0), torch.cat((Labels, Labels), dim = 0))
   
        LTotal =  LossPos1  
        
        LTotal.backward()
        epoch_loss += LTotal.item()

        optimizer.step()
    scheduler.step()
    epoch_losses.append(epoch_loss / (train_step + 1))

    #Plot latent embeddings
    val_loss = 0;pos_sim=[];neg_sim=[];neg_sim_aug=[]
    model.eval()
    for val_step, batch in enumerate(val_loader):
        
        Pos11 = batch['data'].to(device); Pos12 = batch['data_pos'].to(device) 
        with torch.no_grad():
        #predictions
            #PosEmb11, _ = model(Pos11.to(device)); PosEmb12, _ = model(Pos12.to(device))
            PosEmb11= Siamese(Pos11.to(device)); PosEmb12= Siamese(Pos12.to(device))
        #loss
        
        if loss_type =='triplet':
            Neg = batch['data_neg'].to(device); NegEmb, _ = model(Neg.to(device))
            val_loss += LossTr(PosEmb11, PosEmb12, NegEmb ).item()
        else:
            Labels = torch.arange(PosEmb11.shape[0])
            val_loss += LossTr(torch.cat((PosEmb11, PosEmb12), dim = 0),  torch.cat((Labels, Labels), dim = 0)).item()       
            similarity_pos = cosine_similarity(PosEmb11, PosEmb12).cpu().numpy()
            similarity_neg = cosine_similarity(PosEmb11, PosEmb11).cpu().numpy()

            pos_sim.append( np.diag(similarity_pos))
            neg_sim.append (similarity_neg[np.triu_indices_from(similarity_neg, k=1)])
            neg_sim_aug.append (similarity_pos[np.triu_indices_from(similarity_pos, k=1)])
        torch.cuda.empty_cache()
    val_loss /= (val_step+1)
    val_losses.append(val_loss )
    if loss_type !='triplet':
        writer.add_histogram('Positive samples', np.hstack(pos_sim), epoch)
        writer.add_histogram('Negative samples', np.hstack(neg_sim), epoch)
        writer.add_histogram('Negative samples Augmented', np.hstack(neg_sim_aug), epoch)
    writer.add_scalar('Train/Loss', epoch_losses[-1], epoch)
    writer.add_scalar('Val/Loss', val_losses[-1], epoch)

    if (epoch + 1) % save_model_interval == 0 or epoch==0:  
        if multi_gpu:
            torch.save(model.module.state_dict(), ckpt_dir +"/model"+ str(epoch))
        else:
            torch.save(model.state_dict(), ckpt_dir +"/model"+ str(epoch))
    if (epoch >1) and (val_loss < min(val_losses[:-1])):
        if multi_gpu:
            torch.save(model.module.state_dict(), ckpt_dir +"model_best")
        else:
            torch.save(model.state_dict(), ckpt_dir +"model_best")
 