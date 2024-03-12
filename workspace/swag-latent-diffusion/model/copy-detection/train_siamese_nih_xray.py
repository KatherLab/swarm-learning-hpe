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
#sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))

from networks.SiameseNetwork import SiameseNetwork
from pytorch_metric_learning.distances import BatchedDistance, CosineSimilarity
from pytorch_metric_learning.losses import NTXentLoss
from dataset.nih_chest_xray_cl import NIHXRayDataset

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
parser.add_argument('--data_dir', default='/mnt/sds/sd20i001/salman/data/NIHXRay')
parser.add_argument('--n_epochs', type = int,default=150)
parser.add_argument('--batch_size', type = int, default=16)
parser.add_argument('--training_samples', type = int, default=10000)
parser.add_argument('--val_interval', type = int, default=1)
parser.add_argument('--base_lr', type = float, default=10e-4)#
parser.add_argument('--num_channels', type=tuple_type,  default=(64,)*7)#
parser.add_argument('--attention_levels', type=tuple_type,  default=(False, False, False))#
parser.add_argument('--save_model_interval', type = int, default=5)
parser.add_argument('--autoencoder_warm_up_n_epochs', type = int, default=10)
parser.add_argument('--ckpt_dir', type = str, default='/mnt/sds/sd20i001/salman/ckpt/Diffusion-2D-NIHXray/siamese_network_try2/')
parser.add_argument('--details', type = str, default='')
parser.add_argument('--downsample', type = int, default=2)
parser.add_argument('--loss_type', type = str, default='BCEWithLogitsLoss')
parser.add_argument('--triplet_loss_norm', type = int, default=1)
parser.add_argument('--triplet_loss_define_loss', action='store_true')
parser.add_argument('--temperature',type = float, default=0.07)
parser.add_argument('--multi_gpu', action='store_false')
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
train_data = NIHXRayDataset(root_dir= data_dir, split = 'train', training_samples = training_samples , donwsample = downsample)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)

#Load validation data
val_data = NIHXRayDataset(root_dir= data_dir, split = 'val', donwsample = downsample)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)
cosine_similarity = CosineSimilarity()

#Define network
device = torch.device("cuda")
model = SiameseNetwork()
if multi_gpu: model = MyDataParallel(model)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

#Training
optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, eta_min = base_lr/2.0)
LossTr = nn.BCEWithLogitsLoss()
writer = SummaryWriter(log_dir= ckpt_dir)


epoch_losses =[]
val_losses = []
for epoch in range(n_epochs):
    
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
    progress_bar.set_description(f"Epoch {epoch}")    

    epoch_loss = 0
    model.train()
    for train_step, batch in progress_bar:
        #Set gradient to zero
        optimizer.zero_grad()
        #Obtain samples
        Pos11 = batch['data'].to(device); Pos12 = batch['data_pos'].to(device);  Neg = batch['data_neg'].to(device)

        #Obtain predictions
        pred_pos, _, _ = model(Pos11, Pos12 ); pred_neg1,_,_ = model(Pos11, Neg) ; pred_neg2,_,_ = model(Pos12, Neg) 
    

        LossPos = LossTr(pred_pos, torch.zeros_like(pred_pos)) ; LossNeg1 = LossTr(pred_neg1, torch.ones_like(pred_neg1)) ; LossNeg2 = LossTr(pred_neg2, torch.ones_like(pred_neg2)) 
        LTotal =  LossPos + LossNeg1 + LossNeg2
        
        LTotal.backward()
        epoch_loss += LTotal.item()

        optimizer.step()
    scheduler.step()
    epoch_losses.append(epoch_loss / (train_step + 1))

    #Plot latent embeddings
    val_loss = 0;val_pos_loss = 0;val_neg_loss = 0;val_neg_loss_aug = 0; pos_sim=[];neg_sim=[];neg_sim_aug=[]
    model.eval()
    for val_step, batch in enumerate(val_loader):
        
        #Obtain samples
        Pos11 = batch['data'].to(device); Pos12 = batch['data_pos'].to(device);  Neg = batch['data_neg'].to(device)

        with torch.no_grad():
        #predictions
            pred_pos, self_emb, pos_emb = model(Pos11, Pos12 ); pred_neg1,_,neg_emb = model(Pos11, Neg) ; pred_neg2,_,_ = model(Pos12, Neg) 
        
        #loss
        LossPos = LossTr(pred_pos, torch.zeros_like(pred_pos)) 
        LossNeg1 = LossTr(pred_neg1, torch.ones_like(pred_neg1)) ; LossNeg2 = LossTr(pred_neg2, torch.ones_like(pred_neg2)) 

        val_pos_loss+=LossPos.item(); val_neg_loss+=LossNeg1.item(); val_neg_loss_aug+=LossNeg2.item()
        val_loss =  val_pos_loss+ + val_neg_loss + val_neg_loss_aug
        similarity_pos = cosine_similarity(self_emb, pos_emb).cpu().numpy()
        similarity_neg1 = cosine_similarity(self_emb, neg_emb).cpu().numpy()   
        similarity_neg2 = cosine_similarity(pos_emb, neg_emb).cpu().numpy()

        pos_sim.append( np.diagonal(similarity_pos))
        neg_sim.append (np.diagonal(similarity_neg1))
        neg_sim_aug.append (np.diagonal(similarity_neg2))

        torch.cuda.empty_cache()
    val_pos_loss /= (val_step+1); val_neg_loss /= (val_step+1); val_neg_loss_aug /= (val_step+1)
    val_loss /= (val_step+1)
    val_losses.append(val_loss )

    
    writer.add_histogram('Positive samples', np.hstack(pos_sim), epoch)
    writer.add_histogram('Negative samples', np.hstack(neg_sim), epoch)
    writer.add_histogram('Negative samples Augmented', np.hstack(neg_sim_aug), epoch)

    writer.add_scalar('Train/Loss', epoch_losses[-1], epoch)
    writer.add_scalar('Val/Loss', val_losses[-1], epoch)
    writer.add_scalars('Val', {'pos':val_pos_loss,
                                'neg':val_neg_loss,
                                'neg_aug': val_neg_loss_aug}, epoch)

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


 