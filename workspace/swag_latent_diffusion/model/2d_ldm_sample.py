#Load required modules
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from monai import transforms
from monai.apps import MedNISTDataset
from monai.config import print_config
from monai.data import DataLoader, Dataset
from monai.utils import first, set_determinism
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import json
import torchxrayvision as xrv
import torchio as tio

from generative.inferers import LatentDiffusionInferer, DiffusionInferer
from generative.losses.adversarial_loss import PatchAdversarialLoss
from generative.losses.perceptual import PerceptualLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from generative.networks.schedulers import DDPMScheduler

from dataset.nih_chest_xray import NIHXRayDataset
from dataset.fastmri_brain import fastMRIDataset
from generative.metrics.fid import FIDMetric 
from generative.metrics.ms_ssim import MultiScaleSSIMMetric
from torch.utils.tensorboard import SummaryWriter

from onnx2torch import convert

#scla + adjust normal distribution
def normal_adjusted(tensor_shape, mu = 0, sigma = 0.707, shift = 1,  type = 'center'):

    if type == 'center':
        samples = torch.normal(mu, sigma, size=(tensor_shape))
    else:
        samples1 = torch.normal(mu+shift, sigma, size=(tensor_shape)) 
        samples2 = torch.normal(mu-shift, sigma, size=(tensor_shape))
        mask = torch.bernoulli(torch.ones(tensor_shape)*0.5)>0
        samples = samples1 * mask + samples2 * ~mask

    return samples
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
parser.add_argument('--data_dir', default='')
parser.add_argument('--dataset', default='NIHXRay')
parser.add_argument('--n_epochs', type = int,default=150)
parser.add_argument('--epoch_start', type = int,default=0)
parser.add_argument('--batch_size', type = int, default=32)
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
parser.add_argument('--ckpt_dir', type = str, default='')
parser.add_argument('--details', type = str, default='')
parser.add_argument('--downsample', type = int, default=2)
parser.add_argument('--generate_samples', type = int, default=96)

parser.add_argument('--results_dir', type = str, default='')
parser.add_argument('--pretrained_model_dir', type = str, default='')

parser.add_argument('--latent_scaling', type = str, default= 'custom')
parser.add_argument('--custom_scale',type = float, default=0.3)
parser.add_argument('--latent_scaling_batch',type = int, default=16)


parser.add_argument('--multi_gpu', action='store_true')

parser.add_argument('--latent_sampling', type = str, default= '')
parser.add_argument('--shifted_sampling', action='store_true')
parser.add_argument('--shifting',type = float, default=1)
parser.add_argument('--not_load_metrics', action='store_true')
args = parser.parse_args()


#Arguments
data_dir = args.data_dir
dataset = args.dataset
n_epochs = args.n_epochs
epoch_start = args.epoch_start
batch_size =  args.batch_size
training_samples =  args.training_samples
val_interval = args.val_interval

save_model_interval = args.save_model_interval
num_channels_ae = args.num_channels_ae
ckpt_dir = args.ckpt_dir
results_dir = args.results_dir

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

pretrained_model_dir=args.pretrained_model_dir

latent_scaling = args.latent_scaling
latent_sampling = args.latent_sampling
latent_scaling_batch = args.latent_scaling_batch
custom_scale = args.custom_scale
if latent_scaling_batch>batch_size: latent_scaling_batch = batch_size
multi_gpu = args.multi_gpu

shifted_sampling = args.shifted_sampling
shifting = args.shifting
not_load_metrics = args.not_load_metrics
resize = tio.Resize(target_shape=( 1, 224, 224))

preprocessing_pretrained_model = tio.Compose([
    tio.Resize(target_shape=( 1, 224, 224)),
    tio.RescaleIntensity(out_min_max=(-1024, 1024)) 
])

if latent_sampling == 'shifted':
    results_dir+= 'shifted_'+str(int(shifting))+'/'
elif latent_sampling == 'uniform':
    results_dir+= 'uniform/'
elif latent_sampling == 'center' or latent_sampling == 'tails':
    results_dir+=  latent_sampling+'/'

isExist = os.path.exists(results_dir)
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(results_dir)
#Save arguments
with open(results_dir + '/arguments.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2) 


#Load validation data
if 'NIHXRay' in dataset:
    val_data = NIHXRayDataset(root_dir= data_dir, split = 'val', donwsample = downsample,validation_samples=generate_samples)
elif 'fastMRI' in dataset:
    val_data = fastMRIDataset(root_dir= data_dir, split = 'val', validation_samples=generate_samples)

val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps, beta_schedule=beta_schedule, beta_start=beta_start, beta_end=beta_end,prediction_type=prediction_type)
#if multi_gpu: scheduler = torch.nn.DataParallel(scheduler )
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
autoencoderkl = autoencoderkl.to(device);autoencoderkl.eval()

#pre trained model for metric calculation
#pretrained_model = convert(pretrained_model_dir)
pretrained_model = xrv.models.DenseNet(weights="densenet121-res224-nih")
if multi_gpu: pretrained_model = MyDataParallel(pretrained_model) 
pretrained_model.to(device);pretrained_model.eval()
FID = FIDMetric()
MS_SSIM = MultiScaleSSIMMetric(spatial_dims=2)

#epochs = list(range(0,1))+list(range(save_model_interval-1,n_epochs,save_model_interval)) + ['_best_ldm']
epochs = [np.max([epoch_start-1,0])]+list(range(epoch_start+save_model_interval-1,n_epochs,save_model_interval))+ ['_best_ldm']
#epochs.reverse()
epoch_fids = [];epoch_ssims = []
if epoch_start>0 and not(not_load_metrics): 
    epoch_fids = np.load(results_dir+'metrics_saving_interval'+str(save_model_interval)+'.npz')['fid'][:-1].tolist()
    epoch_ssims = np.load(results_dir+'metrics_saving_interval'+str(save_model_interval)+'.npz')['ms_ssim'][:-1].tolist()
#Tensorboard
writer = SummaryWriter(log_dir= results_dir)

for epoch in epochs:
    epoch_fid = 0
    epoch_ssim = 0
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
    unet.load_state_dict(torch.load(ckpt_dir + 'model'+str(epoch)))
    if multi_gpu: unet = MyDataParallel(unet) 
    unet = unet.to(device)
    unet.eval()
    #Set scal factor
    with torch.no_grad():
        with autocast(enabled=True):
            z = autoencoderkl.encode_stage_2_inputs(first(val_loader)['data'][:latent_scaling_batch,:].to(device))
            

    print(f"Batch scale factor {1/torch.std(z)}")
    if latent_scaling == "batch":
        scale_factor = 1 / torch.std(z)
    elif latent_scaling == "custom":
        scale_factor = custom_scale
    else:
        scale_factor = 1    

    print(f"Scaling factor set to {scale_factor}")
    #I'm loading regular inferer instead of latent becuase autoencoder is not worling on multi gpus
    inferer = DiffusionInferer(scheduler)
    #if multi_gpu: inferer= torch.nn.DataParallel(inferer) 

    # Sample images
    sampled_images =[]
    sampling_shape = list(z.shape); sampling_shape[0] = batch_size
    #load validation data and compare with generated samples
    for val_step, batch in enumerate(val_loader, start=1):
        images = batch["data"].to(device)
        with torch.no_grad():
            if latent_sampling =='uniform':
                r1=3;r2=-3
                z = (r1 - r2) * torch.rand(sampling_shape, generator=torch.Generator()) + r2 
            elif latent_sampling == 'center' or latent_sampling == 'tails':
                z = normal_adjusted(sampling_shape, mu = 0, sigma = 0.707, shift = 1.5,  type = latent_sampling)
            elif latent_sampling =='shifted':
                z = normal_adjusted(sampling_shape, mu = shifting, sigma = 0.707, shift = 1.5,  type = 'center')
            else:
                z = torch.randn(sampling_shape, generator=torch.Generator())

            z = z.to(device)    
            scheduler.set_timesteps(num_inference_steps=1000)
            #with autocast(enabled=True):
                # sampled = inferer.sample(
                #     input_noise=z, diffusion_model=unet, scheduler=scheduler, autoencoder_model=autoencoderkl
                # )     
            latent_sampled = inferer.sample(input_noise=z, diffusion_model=unet, scheduler=scheduler)    
            sampled, _,_ = autoencoderkl(first(val_loader)['data'][:latent_scaling_batch,:].to(device),False, latent_sampled/ scale_factor)  
            dummy, _,_ = autoencoderkl(first(val_loader)['data'][:latent_scaling_batch,:].to(device))  
            images_features = pretrained_model.features2(preprocessing_pretrained_model(images.cpu()).to(device) )
            sampled_features = pretrained_model.features2(preprocessing_pretrained_model(sampled.cpu()).to(device) )
        #images_features = F.adaptive_avg_pool2d(images_features, 1).squeeze(-1).squeeze(-1)
        #sampled_features = F.adaptive_avg_pool2d(sampled_features, 1).squeeze(-1).squeeze(-1)
        print(val_step)
        #batch_size = images_features.shape[0]; 
        #feature_size = np.prod(images_features[0,:].shape)
        epoch_fid += FID(images_features , sampled_features)
        indices1 = np.arange(batch_size);np.random.shuffle(indices1)
        indices2 = np.arange(batch_size);np.random.shuffle(indices2)

        epoch_ssim += MS_SSIM(sampled[indices1[-int(batch_size/2):] , :], sampled[indices1[0:int(batch_size/2)], :]).mean()/2
        epoch_ssim += MS_SSIM(sampled[indices2[-int(batch_size/2):] , :], sampled[indices2[0:int(batch_size/2)], :]).mean()/2
        sampled_images.append(np.float16(sampled.cpu().numpy()))
    sampled_images = np.concatenate(sampled_images) 
    np.savez(results_dir+'synthesized_samples_'+str(epoch)+'.npz', synth = sampled_images)
    
    epoch_fids.append(epoch_fid.cpu().numpy()  / (val_step + 1))
    epoch_ssims.append(epoch_ssim.cpu().numpy()  / (val_step + 1))
    np.savez(results_dir+'metrics_saving_interval_'+str(save_model_interval)+'_start_epoch_'+str(epoch_start)+'.npz', fid =epoch_fids, ms_ssim = epoch_ssims ) 
    if epoch =='_best_ldm':
        writer.add_scalar('FID', epoch_fids[-1], n_epochs+10)
        writer.add_scalar('MS-SSIM', epoch_ssims[-1], n_epochs+10)
        writer.add_images('Image Best Val error', sampled.cpu())
    else:
        writer.add_scalar('FID', epoch_fids[-1], epoch)
        writer.add_scalar('MS-SSIM', epoch_ssims[-1], epoch)
        writer.add_images('Image', sampled.cpu(), epoch)
    if epoch==epochs[0]:
        writer.add_images('Val Images', images.cpu())


