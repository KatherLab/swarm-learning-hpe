from odelia.models import BasicClassifier
import monai.networks.nets as nets
import torch
import torch.nn.functional as F

class DenseNet121(BasicClassifier):
    def __init__(
            self,
            in_ch,
            out_ch,
            spatial_dims=3,
            loss=torch.nn.BCEWithLogitsLoss,
            loss_kwargs={},
            optimizer=torch.optim.AdamW,
            optimizer_kwargs={'lr': 1e-4},
            lr_scheduler=None,
            lr_scheduler_kwargs={},
            aucroc_kwargs={"task": "binary"},
            acc_kwargs={"task": "binary"}
    ):
        super().__init__(in_ch, out_ch, spatial_dims, loss, loss_kwargs, optimizer, optimizer_kwargs, lr_scheduler,
                         lr_scheduler_kwargs, aucroc_kwargs, acc_kwargs)
        self.model = nets.DenseNet121(spatial_dims=spatial_dims, in_channels=in_ch, out_channels=out_ch)

    def forward(self, x_in, **kwargs):
        pred_hor = self.model(x_in)
        return pred_hor