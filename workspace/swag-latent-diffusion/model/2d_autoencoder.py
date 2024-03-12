#Load required modules
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from monai import transforms
from monai.apps import MedNISTDataset
from monai.config import print_config
from monai.data import DataLoader, Dataset
from monai.utils import first, set_determinism
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import json

from generative.inferers import LatentDiffusionInferer
from generative.losses.adversarial_loss import PatchAdversarialLoss
from generative.losses.perceptual import PerceptualLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from generative.networks.schedulers import DDPMScheduler

from dataset.nih_chest_xray import NIHXRayDataset
from dataset.nih_chest_xray_subwise import NIHXRayDatasetSubwise
from dataset.fastmri_brain import fastMRIDataset
from torch.utils.tensorboard import SummaryWriter

def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)

#Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='')
parser.add_argument('--dataset', default='NIHXRay')
parser.add_argument('--n_epochs', type = int,default=400)
parser.add_argument('--batch_size', type = int, default=12)
parser.add_argument('--training_samples', type = int, default=10)
parser.add_argument('--val_interval', type = int, default=1)
parser.add_argument('--kl_weight', type = float, default=0.00000001)#
parser.add_argument('--base_lr', type = float, default=0.00005)#
parser.add_argument('--disc_lr', type = float, default=0.0001)#
parser.add_argument('--perceptual_weight', type = float, default=0.002)#
parser.add_argument('--adv_weight', type = float, default=0.005)#
parser.add_argument('--num_channels', type=tuple_type,  default=(64,128,128,128))#
parser.add_argument('--attention_levels', type=tuple_type,  default=(False, False, False,False))#
parser.add_argument('--save_model_interval', type = int, default=5)
parser.add_argument('--autoencoder_warm_up_n_epochs', type = int, default=10)
parser.add_argument('--ckpt_dir', type = str, default='')
parser.add_argument('--details', type = str, default='')
parser.add_argument('--downsample', type = int, default=2)
parser.add_argument('--perceptual_network', type = str, default='squeeze')#
parser.add_argument('--load_checkpoint',  action='store_true')#
parser.add_argument('--augmentation',  action='store_true')#
parser.add_argument('--weighted_sampling',  action='store_true')#
parser.add_argument('--subject_wise',  action='store_true')#
args = parser.parse_args()

#Arguments
data_dir = args.data_dir
dataset = args.dataset
n_epochs = args.n_epochs
batch_size =  args.batch_size
training_samples =  args.training_samples
val_interval = args.val_interval
kl_weight = args.kl_weight
save_model_interval = args.save_model_interval
autoencoder_warm_up_n_epochs = args.autoencoder_warm_up_n_epochs
ckpt_dir = args.ckpt_dir
perceptual_weight = args.perceptual_weight
adv_weight = args.adv_weight 
num_channels = args.num_channels
downsample = args.downsample
base_lr = args.base_lr
disc_lr = args.disc_lr
attention_levels = (False,)*len(num_channels)
perceptual_network = args.perceptual_network
load_checkpoint = args.load_checkpoint
augmentation = args.augmentation
weighted_sampling = args.weighted_sampling
subject_wise = args.subject_wise

isExist = os.path.exists(ckpt_dir)
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(ckpt_dir)
#Save arguments
with open(ckpt_dir + '/arguments.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2) 
#Data
if 'NIHXRay' in dataset:
    if subject_wise:
        train_data = NIHXRayDatasetSubwise(root_dir= data_dir, split = 'train', training_samples = training_samples , donwsample = downsample, augmentation=augmentation)
    else:
        train_data = NIHXRayDataset(root_dir= data_dir, split = 'train', training_samples = training_samples , donwsample = downsample, augmentation=augmentation)
    sample_weight = train_data._get_sampler_weights()
    val_data = NIHXRayDataset(root_dir= data_dir, split = 'val', donwsample = downsample)
elif 'fastMRI' in dataset:
    train_data = fastMRIDataset(root_dir= data_dir, split = 'train', training_samples = training_samples ,  augmentation=augmentation)
    val_data = fastMRIDataset(root_dir= data_dir, split = 'val')
    sample_weight = train_data._get_sampler_weights()

if weighted_sampling:
    sampler = WeightedRandomSampler(weights=sample_weight, num_samples=training_samples , replacement=True);shuffle=None
else:
    sampler = None;shuffle=True
#Loaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, num_workers=4, persistent_workers=True, sampler = sampler)
val_loader = DataLoader(val_data, batch_size=4, shuffle=False)


#Define autoencoder 
device = torch.device("cuda")
print(num_channels)
autoencoderkl = AutoencoderKL(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    num_channels=num_channels,
    latent_channels=3,
    num_res_blocks=2,
    attention_levels=attention_levels,
    with_encoder_nonlocal_attn=False,
    with_decoder_nonlocal_attn=False,
)
autoencoderkl = autoencoderkl.to(device)
#Losses
perceptual_loss = PerceptualLoss(spatial_dims=2, network_type=perceptual_network)
perceptual_loss.to(device)


#Define discriminator
discriminator = PatchDiscriminator(spatial_dims=2, num_layers_d=3, num_channels=64, in_channels=1, out_channels=1)
discriminator = discriminator.to(device)
#Losses
adv_loss = PatchAdversarialLoss(criterion="least_squares")

#Optimizers
optimizer_g = torch.optim.Adam(autoencoderkl.parameters(), lr=base_lr)
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=disc_lr)

# For mixed precision training
scaler_g = torch.cuda.amp.GradScaler()
scaler_d = torch.cuda.amp.GradScaler()

#Tensorboard
writer = SummaryWriter(log_dir= ckpt_dir)

epoch_recon_losses = []
epoch_gen_losses = []
epoch_disc_losses = []
val_recon_losses = []
val_gen_losses = []
val_disc_losses = []
val_perceptual_losses = []
val_sum_losses = []
intermediary_images = []
num_example_images = 4
epoch_start = 0
if load_checkpoint:
    checkpoint = torch.load( ckpt_dir + "checkpoint.pth")
    autoencoderkl.load_state_dict(checkpoint['autoencoderkl'])
    discriminator.load_state_dict(checkpoint['discriminator'])
    optimizer_g.load_state_dict(checkpoint['optimizer_g'])
    optimizer_d.load_state_dict(checkpoint['optimizer_d'])
    epoch_start = checkpoint['epoch']
    epoch_recon_losses = checkpoint['epoch_recon_losses']
    epoch_gen_losses = checkpoint['epoch_gen_losses']
    epoch_disc_losses = checkpoint['epoch_disc_losses']
    val_recon_losses = checkpoint['val_recon_losses']
    val_gen_losses = checkpoint['val_gen_losses']
    val_disc_losses = checkpoint['val_disc_losses']
    val_perceptual_losses = checkpoint['val_perceptual_losses']
    val_sum_losses = checkpoint['val_sum_losses'] 
for epoch in range(epoch_start, n_epochs):
    autoencoderkl.train()
    discriminator.train()
    epoch_loss = 0
    gen_epoch_loss = 0
    disc_epoch_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:
        images = batch["data"].to(device)
        optimizer_g.zero_grad(set_to_none=True)

        with autocast(enabled=True):
            reconstruction, z_mu, z_sigma = autoencoderkl(images)

            recons_loss = F.l1_loss(reconstruction.float(), images.float())
            p_loss = perceptual_loss(reconstruction.float(), images.float())
            kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
            loss_g = recons_loss + (kl_weight * kl_loss) + (perceptual_weight * p_loss)

            if epoch > autoencoder_warm_up_n_epochs:
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                loss_g += adv_weight * generator_loss

        scaler_g.scale(loss_g).backward()
        scaler_g.step(optimizer_g)
        scaler_g.update()

        if epoch > autoencoder_warm_up_n_epochs:
            with autocast(enabled=True):
                optimizer_d.zero_grad(set_to_none=True)

                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

                loss_d = adv_weight * discriminator_loss

            scaler_d.scale(loss_d).backward()
            scaler_d.step(optimizer_d)
            scaler_d.update()

        epoch_loss += recons_loss.item()
        if epoch > autoencoder_warm_up_n_epochs:
            gen_epoch_loss += generator_loss.item()
            disc_epoch_loss += discriminator_loss.item()

        progress_bar.set_postfix(
            {
                "recons_loss": epoch_loss / (step + 1),
                "gen_loss": gen_epoch_loss / (step + 1),
                "disc_loss": disc_epoch_loss / (step + 1),
            }
        )
    epoch_recon_losses.append(epoch_loss / (step + 1))
    epoch_gen_losses.append(gen_epoch_loss / (step + 1))
    epoch_disc_losses.append(disc_epoch_loss / (step + 1))

    if (epoch + 1) % val_interval == 0:
        autoencoderkl.eval()
        discriminator.eval()
        val_recon_loss = 0
        val_g_loss = 0
        val_d_loss = 0
        val_p_loss = 0
        with torch.no_grad():
            for val_step, batch in enumerate(val_loader, start=1):
                images = batch["data"].to(device)

                with autocast(enabled=True):
                    reconstruction, z_mu, z_sigma = autoencoderkl(images)
                    recons_loss = F.l1_loss(images.float(), reconstruction.float())
                    p_loss = perceptual_loss(reconstruction.float(), images.float())
                if epoch > autoencoder_warm_up_n_epochs:
                    logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                    generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)   

                    with autocast(enabled=True):
                        optimizer_d.zero_grad(set_to_none=True)

                        logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                        loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                        logits_real = discriminator(images.contiguous().detach())[-1]
                        loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                        discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

                        loss_d = adv_weight * discriminator_loss                                   

                else:
                    generator_loss = torch.tensor([0]).to(device)
                    loss_d = torch.tensor([0]).to(device)
                    # Get the first reconstruction from the first validation batch for visualisation purposes
                if val_step == 1:
                    val_imgs = reconstruction[:num_example_images].cpu()
                    if epoch == 0:
                        ref_imgs = images[:num_example_images].cpu()

                val_recon_loss += recons_loss.item()
                val_g_loss += generator_loss.item()
                val_p_loss += p_loss.item()
                val_d_loss += loss_d.item()
                

        val_recon_loss /= val_step
        val_g_loss /= val_step
        val_d_loss /= val_step
        val_p_loss /= val_step

        val_sum_loss = perceptual_weight * val_p_loss + adv_weight * val_g_loss + val_recon_loss

        val_recon_losses.append(val_recon_loss)
        val_gen_losses.append(val_g_loss)
        val_disc_losses.append(val_d_loss)
        val_perceptual_losses.append(val_p_loss)
        val_sum_losses.append(val_sum_loss)

        print(f"epoch {epoch + 1} val loss: {val_recon_loss:.4f}")

        checkpoint = {
            "epoch": epoch + 1,
            "autoencoderkl": autoencoderkl.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "epoch_recon_losses":epoch_recon_losses,
            "epoch_gen_losses":epoch_gen_losses,
            "epoch_disc_losses":epoch_disc_losses,
            "val_recon_losses":val_recon_losses,
            "val_gen_losses":val_gen_losses,
            "val_disc_losses":val_disc_losses,
            "val_perceptual_losses":val_perceptual_losses,
            "val_sum_losses":val_sum_losses
        }
        torch.save(checkpoint, ckpt_dir + "checkpoint.pth") 
    if (epoch + 1) % save_model_interval == 0 or epoch==0:    
        torch.save(autoencoderkl.state_dict(), ckpt_dir +"model"+ str(epoch))
        writer.add_images('Val/Images', val_imgs, epoch)
    if  epoch==0:   
        writer.add_images('Val/RefImages', ref_imgs)

    if (epoch >autoencoder_warm_up_n_epochs) and (val_sum_loss < min(val_sum_losses[autoencoder_warm_up_n_epochs:-1])):
        torch.save(autoencoderkl.state_dict(), ckpt_dir +"model_best_ae")
        torch.save(discriminator.state_dict(), ckpt_dir +"model_best_disc")

    writer.add_scalar('Train/Recon', epoch_recon_losses[-1], epoch)
    writer.add_scalar('Train/Gen', epoch_gen_losses[-1], epoch)
    writer.add_scalar('Train/Disc', epoch_disc_losses[-1], epoch)

    writer.add_scalar('Val/Recon', val_recon_losses[-1], epoch)
    writer.add_scalar('Val/Gen', val_gen_losses[-1], epoch)
    writer.add_scalar('Val/Disc', val_disc_losses[-1], epoch)
    writer.add_scalar('Val/Perceptual', val_perceptual_losses[-1], epoch)
    writer.add_scalar('Val/Sum', val_sum_losses[-1], epoch)    

progress_bar.close()

del discriminator
del perceptual_loss
torch.cuda.empty_cache()
