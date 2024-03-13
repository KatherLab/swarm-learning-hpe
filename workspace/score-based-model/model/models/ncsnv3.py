from torch import nn
import torch
from . import get_sigmas
from .layers import *


class NCSNv3Deepest(nn.Module):
    def __init__(self, config):
        super().__init__()

        in_channel = config.data.channels
        out_channel = config.data.channels

        channel = config.model.channel
        channel_multiplier = config.model.channel_multiplier
        n_res_blocks = config.model.n_res_blocks
        attn_strides = config.model.attn_strides
        attn_heads = config.model.attn_heads
        use_affine_time = config.model.use_affine_time
        dropout = config.model.dropout
        fold = config.model.fold

        num_classes = config.data.num_classes

        self.config = config

        self.register_buffer('sigmas', get_sigmas(config))

        self.fold = fold

        time_dim = channel * 4

        n_block = len(channel_multiplier)

        self.time = nn.Sequential(
            TimeEmbedding(channel),
            linear(channel, time_dim),
            Swish(),
            linear(time_dim, time_dim),
        )

        down_layers = [conv2d(in_channel * (fold ** 2), channel, 3, padding=1)]
        feat_channels = [channel]
        in_channel = channel
        for i in range(n_block):
            for _ in range(n_res_blocks):
                channel_mult = channel * channel_multiplier[i]

                down_layers.append(
                    ResBlockWithAttention(
                        in_channel,
                        channel_mult,
                        time_dim,
                        dropout,
                        use_attention=2 ** i in attn_strides,
                        attention_head=attn_heads,
                        use_affine_time=use_affine_time,
                    )
                )

                feat_channels.append(channel_mult)
                in_channel = channel_mult

            if i != n_block - 1:
                down_layers.append(Downsample(in_channel))
                feat_channels.append(in_channel)

        self.down = nn.ModuleList(down_layers)

        self.mid = nn.ModuleList(
            [
                ResBlockWithAttention(
                    in_channel,
                    in_channel,
                    time_dim,
                    dropout=dropout,
                    use_attention=True,
                    attention_head=attn_heads,
                    use_affine_time=use_affine_time,
                ),
                ResBlockWithAttention(
                    in_channel,
                    in_channel,
                    time_dim,
                    dropout=dropout,
                    use_affine_time=use_affine_time,
                ),
            ]
        )

        up_layers = []
        for i in reversed(range(n_block)):
            for _ in range(n_res_blocks + 1):
                channel_mult = channel * channel_multiplier[i]

                up_layers.append(
                    ResBlockWithAttention(
                        in_channel + feat_channels.pop(),
                        channel_mult,
                        time_dim,
                        dropout=dropout,
                        use_attention=2 ** i in attn_strides,
                        attention_head=attn_heads,
                        use_affine_time=use_affine_time,
                    )
                )

                in_channel = channel_mult

            if i != 0:
                up_layers.append(Upsample(in_channel))

        self.up = nn.ModuleList(up_layers)

        self.out = nn.Sequential(
            nn.GroupNorm(32, in_channel),
            Swish(),
            conv2d(in_channel, out_channel * (fold ** 2),
                   3, padding=1, scale=1e-10),
        )

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def forward(self, input, time, label=None):
        time_embed = self.time(time)

        # Add Label to Time Embedding
        if label is not None:
            time_embed += self.label_emb(label)

        feats = []

        out = spatial_fold(input, self.fold)
        for layer in self.down:
            if isinstance(layer, ResBlockWithAttention):
                out = layer(out, time_embed)

            else:
                out = layer(out)

            feats.append(out)

        for layer in self.mid:
            out = layer(out, time_embed)

        for layer in self.up:
            if isinstance(layer, ResBlockWithAttention):
                out = layer(torch.cat((out, feats.pop()), 1), time_embed)
            else:
                out = layer(out)

        out = self.out(out)
        out = spatial_unfold(out, self.fold)

        used_sigmas = self.sigmas[time].view(
            input.shape[0], *([1] * len(input.shape[1:])))

        out = out / used_sigmas

        return out
