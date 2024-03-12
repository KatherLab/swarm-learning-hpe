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
from env_config import load_environment_variables
from generative.inferers import LatentDiffusionInferer
from generative.losses.adversarial_loss import PatchAdversarialLoss
from generative.losses.perceptual import PerceptualLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from generative.networks.schedulers import DDPMScheduler

from dataset.nih_chest_xray import NIHXRayDataset
from dataset.nih_chest_xray_subwise import NIHXRayDatasetSubwise
from dataset.fastmri_brain import fastMRIDataset
from torch.utils.tensorboard import SummaryWriter
from swarmlearning.pyt import SwarmCallback

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
'''
#Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='')
parser.add_argument('--dataset', default='NIHXRay')
parser.add_argument('--n_epochs', type = int,default=100)
parser.add_argument('--batch_size', type = int, default=4)
parser.add_argument('--training_samples', type = int, default=10)
parser.add_argument('--val_interval', type = int, default=1)
parser.add_argument('--num_channels_ae', type=tuple_type,  default=(64, 128, 128, 128))
parser.add_argument('--ae_ckpt', type = str, default ='')

parser.add_argument('--num_channels', type=tuple_type,  default=(256, 512, 768))
parser.add_argument('--num_head_channels', type=tuple_type,  default=(0, 512, 768))

parser.add_argument('--base_lr',type = float, default = 0.000025)
parser.add_argument('--beta_schedule',type = str, default = "scaled_linear")
parser.add_argument('--num_train_timesteps',type = int, default=1000)
parser.add_argument('--beta_start',type = float, default=0.0015)
parser.add_argument('--beta_end',type = float, default=0.0205)
parser.add_argument('--prediction_type',type = str, default="v_prediction")

parser.add_argument('--save_model_interval', type = int, default=50)
parser.add_argument('--save_ckpt_interval', type = int, default=5)
parser.add_argument('--ckpt_dir', type = str, default='')
parser.add_argument('--details', type = str, default='')
parser.add_argument('--downsample', type = int, default=2)
parser.add_argument('--generate_samples', type = int, default=3)

parser.add_argument('--latent_scaling', type = str, default= 'custom')
parser.add_argument('--custom_scale',type = float, default=0.3)
parser.add_argument('--load_checkpoint',  action='store_true')#
parser.add_argument('--multi_gpu', action='store_true')
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

save_model_interval = args.save_model_interval
save_ckpt_interval = args.save_ckpt_interval
num_channels_ae = args.num_channels_ae
ckpt_dir = args.ckpt_dir

downsample = args.downsample
ae_ckpt = args.ae_ckpt
attention_levels_ae = (False,)*len(num_channels_ae)

num_head_channels = args.num_head_channels
num_channels = args.num_channels

base_lr=args.base_lr
beta_schedule=args.beta_schedule
num_train_timesteps=args.num_train_timesteps
beta_start=args.beta_start
beta_end=args.beta_end
prediction_type=args.prediction_type
generate_samples=args.generate_samples

latent_scaling = args.latent_scaling
custom_scale = args.custom_scale
load_checkpoint = args.load_checkpoint

multi_gpu = args.multi_gpu
augmentation = args.augmentation
weighted_sampling = args.weighted_sampling
subject_wise = args.subject_wise
'''

env_vars = load_environment_variables()

ckpt_dir = env_vars["ckpt_dir"]
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
#Save arguments
with open(os.path.join(ckpt_dir, 'arguments.txt'), 'w') as f:
    # Assuming args-like structure is in env_vars or an equivalent has been constructed
    json.dump(env_vars, f, indent=2)
# Data loaders setup
dataset = env_vars['dataset']
data_dir = env_vars['data_dir']
training_samples = env_vars['training_samples']
downsample = env_vars['downsample']
augmentation = env_vars['augmentation']
subject_wise = env_vars['subject_wise']
weighted_sampling = env_vars['weighted_sampling']
num_channels = env_vars['num_channels']
num_channels_ae = env_vars['num_channels_ae']
ae_ckpt = env_vars['ae_ckpt']
batch_size = env_vars['batch_size']
attention_levels_ae = env_vars['attention_levels_ae']
multi_gpu = env_vars['multi_gpu']
num_channels_ae = env_vars['num_channels_ae']
latent_scaling = env_vars['latent_scaling']
custom_scale = env_vars['custom_scale']
num_head_channels = env_vars['num_head_channels']
num_train_timesteps = env_vars['num_train_timesteps']
n_epochs = env_vars['n_epochs']
val_interval = env_vars['val_interval']
save_model_interval = env_vars['save_model_interval']
save_ckpt_interval = env_vars['save_ckpt_interval']
n_epochs = env_vars['n_epochs']
ckpt_dir = env_vars['ckpt_dir']
base_lr = env_vars['base_lr']
generate_samples = env_vars['generate_samples']


if 'NIHXRay' in dataset:
    if subject_wise:
        train_data = NIHXRayDatasetSubwise(root_dir=data_dir, split='train', training_samples=training_samples, downsample=downsample, augmentation=augmentation)
    else:
        train_data = NIHXRayDataset(root_dir=data_dir, split='train', training_samples=training_samples, downsample=downsample, augmentation=augmentation)
    sample_weight = train_data._get_sampler_weights()
    val_data = NIHXRayDataset(root_dir=data_dir, split='val', downsample=downsample)
elif 'fastMRI' in dataset:
    train_data = fastMRIDataset(root_dir=data_dir, split='train', training_samples=training_samples, augmentation=augmentation)
    val_data = fastMRIDataset(root_dir=data_dir, split='val')
    sample_weight = train_data._get_sampler_weights()

if weighted_sampling:
    sampler = WeightedRandomSampler(weights=sample_weight, num_samples=training_samples, replacement=True)
    shuffle = None
else:
    sampler = None
    shuffle = True
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
    num_channels=num_channels_ae,
    latent_channels=3,
    num_res_blocks=2,
    attention_levels=attention_levels_ae,
    with_encoder_nonlocal_attn=False,
    with_decoder_nonlocal_attn=False,
)

#Load auto encoder 
autoencoderkl.load_state_dict(torch.load(ae_ckpt))
if multi_gpu: autoencoderkl = MyDataParallel(autoencoderkl)
autoencoderkl = autoencoderkl.to(device)
#Set scal factor
with torch.no_grad():
    with autocast(enabled=True):
        z = autoencoderkl.encode_stage_2_inputs(first(train_loader)['data'][:8,:].to(device))
#perform scaling
if latent_scaling == "batch":
    scale_factor = 1 / torch.std(z)
elif latent_scaling == "custom":
    scale_factor = custom_scale
else:
    scale_factor = 1    
print(f"Scaling factor set to {scale_factor}")
#DDPM
unet = DiffusionModelUNet(
    spatial_dims=2,
    in_channels=3,
    out_channels=3,
    num_res_blocks=2,
    num_channels=num_channels,
    attention_levels=(False, True, True),
    num_head_channels=num_head_channels,
)

epoch_losses = []
val_losses = []
scaler = GradScaler()
num_example_images = 4
epoch_start = 0


scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps, beta_schedule=beta_schedule, beta_start=beta_start, beta_end=beta_end, prediction_type=prediction_type)

if multi_gpu: unet = MyDataParallel(unet) 
unet = unet.to(device)
optimizer = torch.optim.Adam(unet.parameters(), lr=base_lr)
if load_checkpoint:
    checkpoint = torch.load( ckpt_dir + "checkpoint.pth")
    optimizer.load_state_dict(checkpoint['optimizer'])
    if multi_gpu: 
        unet.module.load_state_dict(checkpoint['diffusion'])
    else: 
        unet.load_state_dict(checkpoint['diffusion'])
    epoch_start = checkpoint['epoch']
    val_losses = checkpoint['val_losses']
    epoch_losses = checkpoint['epoch_losses']
    print("Successfully loaded "+ckpt_dir + "checkpoint.pth")



inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)
#Tensorboard
writer = SummaryWriter(log_dir= ckpt_dir)

# swarm callback definition
swSyncInterval = env_vars['sync_frequency']
min_peers = env_vars['min_peers']
max_epochs = env_vars['max_epochs']
swarmCallback = SwarmCallback(syncFrequency=swSyncInterval,
                              minPeers=min_peers,
                              useAdaptiveSync=False,
                              model=unet,
                              totalEpochs=max_epochs,
                              )

swarmCallback.on_train_begin()

for epoch in range(epoch_start,n_epochs):
    unet.train()
    autoencoderkl.eval()
    epoch_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:
        images = batch["data"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=True):

            #z_mu, z_sigma = autoencoderkl.encode(images)
            #z = autoencoderkl.sampling(z_mu, z_sigma,)
            
            with torch.no_grad():
                _, z_mu, z_sigma, z  = autoencoderkl(images, sampled = True)
                e = z * scale_factor
            
            noise = torch.randn_like(z).to(device)

            timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps, (z.shape[0],), device=z.device).long()
            noisy_e = scheduler.add_noise(original_samples=e, noise=noise, timesteps=timesteps)
            noise_pred = unet(x=noisy_e, timesteps=timesteps)

            if scheduler.prediction_type == "v_prediction":
                # Use v-prediction parameterization
                target = scheduler.get_velocity(e, noise, timesteps)
            elif scheduler.prediction_type == "epsilon":
                target = noise   

            loss = F.mse_loss(noise_pred.float(), target.float())
        swarmCallback.on_batch_end()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

        progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
    epoch_losses.append(epoch_loss / (step + 1))

    if (epoch + 1) % val_interval == 0:
        unet.eval()
        val_loss = 0
        with torch.no_grad():
            for val_step, batch in enumerate(val_loader, start=1):
                images = batch["data"].to(device)

                with autocast(enabled=True):
                    #z_mu, z_sigma = autoencoderkl.encode(images)
                    #z = autoencoderkl.sampling(z_mu, z_sigma)
                    _, z_mu, z_sigma, z  = autoencoderkl(images, sampled = True)
                    e = z * scale_factor

                    noise = torch.randn_like(z).to(device)
                    timesteps = torch.randint(
                        0, inferer.scheduler.num_train_timesteps, (z.shape[0],), device=z.device
                    ).long()

                    noisy_e = scheduler.add_noise(original_samples=e, noise=noise, timesteps=timesteps)
                    noise_pred = unet(x=noisy_e, timesteps=timesteps)

                    if scheduler.prediction_type == "v_prediction":
                        # Use v-prediction parameterization
                        target = scheduler.get_velocity(e, noise, timesteps)
                    elif scheduler.prediction_type == "epsilon":
                        target = noise   

                    loss = F.mse_loss(noise_pred.float(), target.float())
                    

                val_loss += loss.item()
        val_loss /= val_step

        if (epoch >1) and (val_loss < min(val_losses[:-1])):
            swarmCallback.on_train_end()
            torch.save(unet.module.state_dict(), ckpt_dir +"model_best_ldm")
            torch.save(autoencoderkl.module.state_dict(), ckpt_dir +"model_best_ae")
        val_losses.append(val_loss)
        print(f"Epoch {epoch} val loss: {val_loss:.4f}")

        # Sampling image during training
        sampling_shape = list(z.shape); sampling_shape[0] = generate_samples
        z = torch.randn(sampling_shape, generator=torch.Generator().manual_seed(2023))
        z = z.to(device)
        scheduler.set_timesteps(num_inference_steps=1000)
        with autocast(enabled=True):
            decoded = inferer.sample(
                input_noise=z, diffusion_model=unet, scheduler=scheduler, autoencoder_model=autoencoderkl
            )
        if (epoch + 1) % save_model_interval == 0 or epoch==0:    
            torch.save(unet.module.state_dict(), ckpt_dir +"model"+ str(epoch))
        if (epoch + 1) % save_ckpt_interval == 0 or epoch==0:        
            checkpoint = {
                "epoch": epoch + 1,
                "diffusion": unet.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_losses": val_losses,
                "epoch_losses": epoch_losses
            }
            torch.save(checkpoint, ckpt_dir + "checkpoint.pth")            

        writer.add_scalar('Train/Loss', epoch_losses[-1], epoch)
        writer.add_scalar('Val/Loss', val_losses[-1], epoch)
        writer.add_images('Image', decoded.cpu(), epoch)
progress_bar.close()

