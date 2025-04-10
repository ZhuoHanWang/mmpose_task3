# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Optional, Sequence, Tuple

import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.model import BaseModule
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm

from mmpose.registry import MODELS
from mmpose.utils.typing import ConfigType
from ..utils import CSPLayer
from .csp_darknet import SPPBottleneck


@MODELS.register_module()
class CSPNeXt(BaseModule):
    """CSPNeXt backbone used in RTMDet.

    Args:
        arch (str): Architecture of CSPNeXt, from {P5, P6}.
            Defaults to P5.
        expand_ratio (float): Ratio to adjust the number of channels of the
            hidden layer. Defaults to 0.5.
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        out_indices (Sequence[int]): Output from which stages.
            Defaults to (2, 3, 4).
        frozen_stages (int): Stages to be frozen (stop grad and set eval
            mode). -1 means not freezing any parameters. Defaults to -1.
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Defaults to False.
        arch_ovewrite (list): Overwrite default arch settings.
            Defaults to None.
        spp_kernel_sizes: (tuple[int]): Sequential of kernel sizes of SPP
            layers. Defaults to (5, 9, 13).
        channel_attention (bool): Whether to add channel attention in each
            stage. Defaults to True.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): Dictionary to construct and
            config norm layer. Defaults to dict(type='BN', requires_grad=True).
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Defaults to dict(type='SiLU').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        init_cfg (:obj:`ConfigDict` or dict or list[dict] or
            list[:obj:`ConfigDict`]): Initialization config dict.
    """
    # From left to right:
    # in_channels, out_channels, num_blocks, add_identity, use_spp
    # 设置的架构是cspnext部分的架构，不包括stem部分
    arch_settings = {
        # P5架构: 4个stage的标准版本
        'P5': [
            # [输入通道数, 输出通道数, CSP中的block数量, 是否使用残差连接, 是否使用SPP层]
            # stage0: stem部分，不属于任何stage
            [64, 128, 3, True, False],    # stage1: 浅层特征提取
            [128, 256, 6, True, False],   # stage2: 中层特征提取，使用更多block
            [256, 512, 6, True, False],   # stage3 (C3): 深层特征提取，检测中小物体
            [512, 1024, 3, False, True]   # stage4 (C4): 最深特征，加入SPP层，检测大物体
        ],
        
        # P6架构: 5个stage的加深版本
        'P6': [
            # [输入通道数, 输出通道数, CSP中的block数量, 是否使用残差连接, 是否使用SPP层]
            # stage0: stem部分，不属于任何stage
            [64, 128, 3, True, False],     # stage1: 与P5相同的浅层特征
            [128, 256, 6, True, False],    # stage2: 与P5相同的中层特征
            [256, 512, 6, True, False],    # stage3: 深层特征
            [512, 768, 3, True, False],    # stage4: 新增的特征层，通道数不翻倍
            [768, 1024, 3, False, True]    # stage5 (C5): 最深特征，同样使用SPP
        ]
    }

    def __init__(
        self,
        arch: str = 'P5', # CSPNeXt的架构，P5或P6
        deepen_factor: float = 1.0, # CSPNeXt的加深因子，控制cspnext瓶颈模块的深度
        widen_factor: float = 1.0,  # 加宽因子，控制通道的大小，通道越大越宽
        out_indices: Sequence[int] = (2, 3, 4), # 这个值确实是索引值，但是由于还会添加stem作为第一个stage，所以索引值和次序值是一样的，也即是默认的P5架构会选择C3，C4作为输出
        frozen_stages: int = -1,
        use_depthwise: bool = False,     # stem和csplayer是否使用深度可分离卷积
        expand_ratio: float = 0.5,       # 用于控制瓶颈模块中隐藏层的通道数相对于输入通道数的比例
        arch_ovewrite: dict = None,      # 用于外部传参控制模型架构参数，覆盖默认的arch_settings 
        spp_kernel_sizes: Sequence[int] = (5, 9, 13), # 注意这里是多个池化核的参数，是5*5，9*9，13*13
        channel_attention: bool = True,
        conv_cfg: Optional[ConfigType] = None, # 卷积层的配置
        norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),   # 归一化层的配置
        act_cfg: ConfigType = dict(type='SiLU'),    # 激活函数的配置
        norm_eval: bool = False,    # 是否设置归一化层为评估模式
        init_cfg: Optional[ConfigType] = dict( # 初始化权重的配置，这是父类BaseModule的参数，会自动初始化每个stage的权重
            type='Kaiming',
            layer='Conv2d',
            a=math.sqrt(5),
            distribution='uniform',
            mode='fan_in',
            nonlinearity='leaky_relu')
    ) -> None:
        super().__init__(init_cfg=init_cfg) # 通过继承自base_module.py的BaseModule类，init_cfg将自动初始化每个stage的权重
        arch_setting = self.arch_settings[arch]
        if arch_ovewrite:
            arch_setting = arch_ovewrite
        assert set(out_indices).issubset(
            i for i in range(len(arch_setting) + 1))
        if frozen_stages not in range(-1, len(arch_setting) + 1):
            raise ValueError('frozen_stages must be in range(-1, '
                             'len(arch_setting) + 1). But received '
                             f'{frozen_stages}')

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.use_depthwise = use_depthwise
        self.norm_eval = norm_eval
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule    # stem部分使用深度可分离卷积

        # stem是整个模型的基础网络部分，与cspnext无关，stem不属于任何stage
        # stem加宽了通道，提取更丰富的特征，让cspblock可以减小深度，提高计算效率
        self.stem = nn.Sequential(
            ConvModule(
                3,      # 输入通道数
                int(arch_setting[0][0] * widen_factor // 2),    # 输出通道数，widen_factor是加宽因子，控制通道的大小，通道越大越宽
                3,      # 卷积核大小
                padding=1,
                stride=2,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                int(arch_setting[0][0] * widen_factor // 2),
                int(arch_setting[0][0] * widen_factor // 2),
                3,
                padding=1,
                stride=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                int(arch_setting[0][0] * widen_factor // 2),
                int(arch_setting[0][0] * widen_factor),
                3,
                padding=1,
                stride=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))
        self.layers = ['stem'] # stem作为根基，接入cspnext的stage

        # 每个stage都有，卷积层、SPP 模块、CSP 模块
        # 开始初始化cspnext的stage
        # i是索引，所以stage=i+1
        for i, (in_channels, out_channels, num_blocks, add_identity,
                use_spp) in enumerate(arch_setting):
            in_channels = int(in_channels * widen_factor)
            out_channels = int(out_channels * widen_factor)
            num_blocks = max(round(num_blocks * deepen_factor), 1) # 使用了加深因子，num_block控制cspnext瓶颈模块的深度
            stage = []
            # 添加降采样卷积层，不是cspnext的一部分
            conv_layer = conv(
                in_channels, # 本stage的输入通道数，arch_settings的第一个元素的输入通道数
                out_channels, # 中间通道数
                3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            stage.append(conv_layer)
            # 论文中说的多尺度特征金字塔
            # spp只会在最后一个stage使用（arch_settings中设置的），spp是多尺度特征提取（不是论文中的多层次特征提取，不体现在这类）
            # spp使用了spp_kernel_sizes的三个递增的池化核，卷积核越大，感受野越大，最后会通过1*1卷积核融合4（三个池化核加上原始的）个通道（1*1卷积核的数字作用于增加的通道，相当于给每个新增通道权重）
            if use_spp:
                spp = SPPBottleneck(
                    out_channels,   # stage的中间通道数
                    out_channels,   # stage的中间通道数
                    kernel_sizes=spp_kernel_sizes,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
                stage.append(spp)
            # csp的核心部分，是瓶颈模块，瓶颈模块的意思，降维，计算，再升维度，论文图3b是csp_layer中的block部分
            csp_layer = CSPLayer(
                out_channels,   # stage的中间通道数
                out_channels,   # stage的输出通道数，arch_settings的第二个元素的输出通道数
                num_blocks=num_blocks,
                add_identity=add_identity,
                use_depthwise=use_depthwise,    # csp使用深度卷积
                use_cspnext_block=True,
                expand_ratio=expand_ratio,
                channel_attention=channel_attention,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            stage.append(csp_layer)
            # 把stage的所有层打包成一个stage，通过add_module注册成为属性，名称索引为stagei+1
            self.add_module(f'stage{i + 1}', nn.Sequential(*stage))
            # layer是stage的索引，所以layer=i+1
            self.layers.append(f'stage{i + 1}')
        # P5 self.layers = ['stem', 'stage1', 'stage2', 'stage3', 'stage4']
        # P6 self.layers = ['stem', 'stage1', 'stage2', 'stage3', 'stage4', 'stage5']
    def _freeze_stages(self) -> None:
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                m = getattr(self, self.layers[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True) -> None:
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    # 选择哪些stage（层次）作为模型的输出，这里是论文中所说的多层次特征提取
    def forward(self, x: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
