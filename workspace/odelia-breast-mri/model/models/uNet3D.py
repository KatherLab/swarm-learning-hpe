from .base_model import BasicClassifier
import monai.networks.nets as nets
import torch
import torch.nn.functional as F

class UNet3D(BasicClassifier):
    def __init__(
            self,
            in_ch,
            out_ch,
            spatial_dims=3,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
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
        self.model = nets.UNet(
            dimensions=spatial_dims,
            in_channels=in_ch,
            out_channels=out_ch,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units
        )

    def forward(self, x_in, **kwargs):
        pred_hor = self.model(x_in)
        return pred_hor

    def _generate_predictions(self, source):
        return self.forward(source)

    def _step(self, batch, batch_idx, phase, optimizer_idx=0):
        source, target = batch['source'], batch['target']

        if phase == "train":
            pred = self._generate_predictions(source)
        elif phase == "val":
            pred = self._generate_predictions(source)
        else:
            raise ValueError(f"Invalid phase: {phase}")

        target = target.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(pred).float()  # Cast target to float
        loss = self.loss(pred, target)


        logging_dict = {f"{phase}_loss": loss}

        if phase == "val":
            logging_dict["y_true"] = target
            logging_dict["y_pred"] = pred

        logging_dict = {k: v.mean() for k, v in logging_dict.items()}  # Add this line before logging
        self.log_dict(logging_dict, on_step=(phase == "train"), on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        return self._step(batch, batch_idx, "val", optimizer_idx)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        return self._step(batch, batch_idx, "train", optimizer_idx)
    # def training_step(self, batch, batch_idx):
    # source, target = batch['source'], batch['target']
    # y_hat = self(source)
    # loss = F.cross_entropy(y_hat, target)
    # return loss
