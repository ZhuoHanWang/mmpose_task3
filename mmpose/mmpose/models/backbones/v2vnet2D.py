import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmpose.registry import MODELS
from .base_backbone import BaseBackbone


class Basic2DBlock(BaseModule):
    """A basic 2D convolutional block."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN2d'),
                 init_cfg=None):
        super(Basic2DBlock, self).__init__(init_cfg=init_cfg)
        self.block = ConvModule(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=((kernel_size - 1) // 2),
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            bias=True)

    def forward(self, x):
        """Forward function."""
        return self.block(x)


class Res2DBlock(BaseModule):
    """A residual 2D convolutional block."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN2d'),
                 init_cfg=None):
        super(Res2DBlock, self).__init__(init_cfg=init_cfg)
        self.res_branch = nn.Sequential(
            ConvModule(
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=((kernel_size - 1) // 2),
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                bias=True),
            ConvModule(
                out_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=((kernel_size - 1) // 2),
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None,
                bias=True))

        if in_channels == out_channels:
            self.skip_con = nn.Sequential()
        else:
            self.skip_con = ConvModule(
                in_channels,
                out_channels,
                1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None,
                bias=True)

    def forward(self, x):
        """Forward function."""
        res = self.res_branch(x)
        skip = self.skip_con(x)
        return F.relu(res + skip, True)


class Pool2DBlock(BaseModule):
    """A 2D max-pool block."""

    def __init__(self, pool_size):
        super(Pool2DBlock, self).__init__()
        self.pool_size = pool_size

    def forward(self, x):
        """Forward function."""
        return F.max_pool2d(
            x, kernel_size=self.pool_size, stride=self.pool_size)


class Upsample2DBlock(BaseModule):
    """A 2D upsample block."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=2,
                 stride=2,
                 init_cfg=None):
        super(Upsample2DBlock, self).__init__(init_cfg=init_cfg)
        assert kernel_size == 2
        assert stride == 2
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
                output_padding=0), nn.BatchNorm2d(out_channels), nn.ReLU(True))

    def forward(self, x):
        """Forward function."""
        return self.block(x)


class EncoderDecoder2D(BaseModule):
    """An encoder-decoder block for 2D data."""

    def __init__(self, in_channels=32, init_cfg=None):
        super(EncoderDecoder2D, self).__init__(init_cfg=init_cfg)

        self.encoder_pool1 = Pool2DBlock(2)
        self.encoder_res1 = Res2DBlock(in_channels, in_channels * 2)
        self.encoder_pool2 = Pool2DBlock(2)
        self.encoder_res2 = Res2DBlock(in_channels * 2, in_channels * 4)

        self.mid_res = Res2DBlock(in_channels * 4, in_channels * 4)

        self.decoder_res2 = Res2DBlock(in_channels * 4, in_channels * 4)
        self.decoder_upsample2 = Upsample2DBlock(in_channels * 4,
                                                 in_channels * 2, 2, 2)
        self.decoder_res1 = Res2DBlock(in_channels * 2, in_channels * 2)
        self.decoder_upsample1 = Upsample2DBlock(in_channels * 2, in_channels,
                                                 2, 2)

        self.skip_res1 = Res2DBlock(in_channels, in_channels)
        self.skip_res2 = Res2DBlock(in_channels * 2, in_channels * 2)

    def forward(self, x):
        """Forward function."""
        skip_x1 = self.skip_res1(x)
        x = self.encoder_pool1(x)
        x = self.encoder_res1(x)

        skip_x2 = self.skip_res2(x)
        x = self.encoder_pool2(x)
        x = self.encoder_res2(x)

        x = self.mid_res(x)

        x = self.decoder_res2(x)
        x = self.decoder_upsample2(x)
        x = x + skip_x2

        x = self.decoder_res1(x)
        x = self.decoder_upsample1(x)
        x = x + skip_x1

        return x


@MODELS.register_module()
class V2VNet2D(BaseBackbone):
    """V2VNet for 2D data."""

    def __init__(self,
                 input_channels,
                 output_channels,
                 mid_channels=32,
                 init_cfg=dict(
                     type='Normal',
                     std=0.001,
                     layer=['Conv2d', 'ConvTranspose2d'])):
        super(V2VNet2D, self).__init__(init_cfg=init_cfg)

        self.front_layers = nn.Sequential(
            Basic2DBlock(input_channels, mid_channels // 2, 7),
            Res2DBlock(mid_channels // 2, mid_channels),
        )

        self.encoder_decoder = EncoderDecoder2D(in_channels=mid_channels)

        self.output_layer = nn.Conv2d(
            mid_channels, output_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """Forward function."""
        x = self.front_layers(x)
        x = self.encoder_decoder(x)
        x = self.output_layer(x)

        return (x, )
