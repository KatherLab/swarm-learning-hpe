from .base_model import BasicClassifier
import torch
import torch.nn.functional as F
import timm

class VisionTransformer(BasicClassifier):
    def __init__(
        self,
        in_ch,
        out_ch,
        spatial_dims=3,
        model_name='vit_base_patch16_224',
        pretrained=False,
        loss=torch.nn.BCEWithLogitsLoss,
        loss_kwargs={},
        optimizer=torch.optim.AdamW,
        optimizer_kwargs={'lr': 1e-4},
        lr_scheduler=None,
        lr_scheduler_kwargs={},
        aucroc_kwargs={"task": "binary"},
        acc_kwargs={"task": "binary"}
    ):
        super().__init__(in_ch, out_ch, spatial_dims, loss, loss_kwargs, optimizer, optimizer_kwargs, lr_scheduler, lr_scheduler_kwargs, aucroc_kwargs, acc_kwargs)
        self.model = VisionTransformer3D(model_name, pretrained=pretrained, in_chans=in_ch, num_classes=out_ch)

    def forward(self, x_in, **kwargs):
        print(x_in.shape)
        pred_hor = self.model(x_in)
        print(pred_hor.shape)
        pred_hor = self.model(x_in)
        return pred_hor

    #def training_step(self, batch, batch_idx):
        #source, target = batch['source'], batch['target']
        #y_hat = self(source)
        #loss = F.cross_entropy(y_hat, target)
        #return loss

import torch.nn as nn

class PatchEmbed3D(nn.Module):
    def __init__(self, in_chans, embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2)
        )

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = self.proj(x)
        x = x.transpose(1, 2)
        return x
from timm.models.vision_transformer import VisionTransformer

class VisionTransformer3D(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        in_chans = kwargs.get("in_chans", 3)
        embed_dim = kwargs.get("embed_dim", 768)
        patch_size = kwargs.get("patch_size", (2, 16, 16))
        self.patch_embed = PatchEmbed3D(in_chans, embed_dim, patch_size)
