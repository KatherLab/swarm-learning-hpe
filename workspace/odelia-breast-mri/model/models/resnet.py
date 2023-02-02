import logging

from odelia.models import BasicClassifier
import monai.networks.nets as nets
import torch 
import torch.nn.functional as F
from swarmlearning.pyt import SwarmCallback

# ------------ Swarm Callback ------------ #!
swarmCallback = SwarmCallback(syncFrequency=128,
                              minPeers=2,
                              useAdaptiveSync=True,
                              adsValData=ds_val,
                              adsValBatchSize=4, )
swarmCallback.logger.setLevel(logging.DEBUG)

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
        loss=torch.nn.CrossEntropyLoss, 
        loss_kwargs={}, 
        optimizer=torch.optim.AdamW, 
        optimizer_kwargs={}, 
        lr_scheduler=None, 
        lr_scheduler_kwargs={}
    ):
        super().__init__(in_ch, out_ch, spatial_dims, loss, loss_kwargs, optimizer, optimizer_kwargs, lr_scheduler, lr_scheduler_kwargs)
        self.model = nets.ResNet(block, layers, block_inplanes, spatial_dims, in_ch, 7, 1, False, 'B', 1.0, out_ch, feed_forward, True)
   
    def forward(self, x_in, **kwargs):
        pred_hor = self.model(x_in)
        return pred_hor

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        swarmCallback.on_batch_end()
        return loss