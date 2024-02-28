from .base_model import BasicClassifier
import torch
import torch.nn.functional as F
import timm


class EfficientNet(BasicClassifier):
    def __init__(
            self,
            in_ch,
            out_ch,
            spatial_dims=3,
            model_name='efficientnet_l2',
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
        super().__init__(in_ch, out_ch, spatial_dims, loss, loss_kwargs, optimizer, optimizer_kwargs, lr_scheduler,
                         lr_scheduler_kwargs, aucroc_kwargs, acc_kwargs)
        self.model = timm.create_model(model_name, pretrained=pretrained, in_chans=in_ch, num_classes=out_ch)

    def forward(self, x_in, **kwargs):
        batch_size, _, num_slices, height, width = x_in.shape
        x_in = x_in.view(batch_size * num_slices, 1, height,
                         width)  # Reshape to [batch_size * num_slices, 1, height, width]

        pred_hor = self.model(x_in)  # Process each slice with EfficientNet

        # Reshape the output back to [batch_size, num_slices, out_ch]
        out_ch = pred_hor.shape[1]
        pred_hor = pred_hor.view(batch_size, num_slices, out_ch)

        # Combine the results from each slice (e.g., by averaging or max-pooling)
        combined_pred = torch.mean(pred_hor, dim=1)

        return combined_pred

    # def training_step(self, batch, batch_idx):
    # source, target = batch['source'], batch['target']
    # y_hat = self(source)
    # loss = F.cross_entropy(y_hat, target)
    # return loss


from .base_model import BasicClassifier
import monai.networks.nets as nets
import torch
class EfficientNet3D(BasicClassifier):
    def __init__(
            self,
            in_ch,
            out_ch,
            spatial_dims=3,
            blocks_args_str=None,
            width_coefficient=1.0,
            depth_coefficient=1.0,
            dropout_rate=0.2,
            image_size=224,
            norm=('batch', {'eps': 0.001, 'momentum': 0.01}),
            drop_connect_rate=0.2,
            depth_divisor=8,
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
        if blocks_args_str is None:
            blocks_args_str = [
                "r1_k3_s11_e1_i32_o16_se0.25",
                "r2_k3_s22_e6_i16_o24_se0.25",
                "r2_k5_s22_e6_i24_o40_se0.25",
                "r3_k3_s22_e6_i40_o80_se0.25",
                "r3_k5_s11_e6_i80_o112_se0.25",
                "r4_k5_s22_e6_i112_o192_se0.25",
                "r1_k3_s11_e6_i192_o320_se0.25"]
        self.model = nets.EfficientNet(blocks_args_str, spatial_dims, in_ch, out_ch,
                                       width_coefficient, depth_coefficient, dropout_rate,
                                       image_size, norm, drop_connect_rate, depth_divisor)

    def forward(self, x_in, **kwargs):
        pred_hor = self.model(x_in)
        return pred_hor

from .base_model import BasicClassifier
import monai.networks.nets as nets
import torch
import torch.nn.functional as F

class EfficientNet3Db7(BasicClassifier):
    def __init__(
            self,
            in_ch,
            out_ch,
            spatial_dims=3,
            blocks_args_str=None,
            width_coefficient=1.0,
            depth_coefficient=1.0,
            dropout_rate=0.2,
            image_size=224,
            norm=('batch', {'eps': 0.001, 'momentum': 0.01}),
            drop_connect_rate=0.2,
            depth_divisor=8,
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
        if blocks_args_str is None:
            blocks_args_str = [
                "r1_k3_s11_e1_i32_o32_se0.25",
                "r4_k3_s22_e6_i32_o48_se0.25",
                "r4_k5_s22_e6_i48_o80_se0.25",
                "r4_k3_s22_e6_i80_o160_se0.25",
                "r6_k5_s11_e6_i160_o256_se0.25",
                "r6_k5_s22_e6_i256_o384_se0.25",
                "r3_k3_s11_e6_i384_o640_se0.25",
            ]

        self.model = nets.EfficientNet(blocks_args_str, spatial_dims, in_ch, out_ch,
                                       width_coefficient, depth_coefficient, dropout_rate,
                                       image_size, norm, drop_connect_rate, depth_divisor)

    def forward(self, x_in, **kwargs):
        pred_hor = self.model(x_in)
        return pred_hor
