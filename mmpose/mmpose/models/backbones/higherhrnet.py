# Copyright (c) OpenMMLab. All rights reserved.
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from mmpose.registry import MODELS
from .base_backbone import BaseBackbone
from .resnet import BasicBlock, Bottleneck, get_expansion

class ScaleAwareModule(BaseModule):
    """Scale-aware feature processing module."""
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg=dict(type='BN')):
        super().__init__()
        
        self.scale_conv = nn.Conv2d(
            in_channels, out_channels, 3, 1, 1)
        self.scale_conv2 = nn.Conv2d(
            out_channels, out_channels, 3, 1, 1)
        self.norm = build_norm_layer(norm_cfg, out_channels)[1]
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.scale_conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.scale_conv2(x)
        return x

@MODELS.register_module()
class HigherHRNet(BaseBackbone):
    """HigherHRNet backbone (CVPR 2020)."""

    def __init__(self,
                 extra,
                 in_channels=3,
                 num_joints=17,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 norm_eval=False,
                 with_cp=False,
                 frozen_stages=-1,
                 init_cfg=None,
                 multiscale_output=True):
        super().__init__(init_cfg=init_cfg)
        
        self.in_channels = in_channels  # 记录输入通道
        self.extra = extra
        self.num_joints = num_joints
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.frozen_stages = frozen_stages
        self.multiscale_output = multiscale_output
        # 添加channels属性
        self.channels = self._get_channels(extra)
        # 论文标准结构初始化
        self._make_stem_layer()  # 先构建stem层
        self._make_stages()      # 再构建stage层
        
        # 尺度感知模块
        self.scale_aware_modules = nn.ModuleList([
            ScaleAwareModule(c, c, norm_cfg)
            for c in self.channels
        ])
        
        # 多分辨率预测头
        self.heads = nn.ModuleList([
            nn.ModuleDict({
                'heatmap': nn.Conv2d(c, num_joints, 1),
                'tagmap': nn.Conv2d(c, num_joints, 1)
            }) for c in self.channels
        ])
        
        # 权重初始化
        self.init_weights()

    def _get_channels(self, extra):
        """从配置中获取各stage的通道数"""
        channels = []
        for stage in ['stage1', 'stage2', 'stage3', 'stage4']:
            if stage in extra:
                channels.extend(extra[stage]['num_channels'])
        return channels
    
    def _make_stem_layer(self):
        """构建论文标准stem层结构"""
        # 第一个3x3卷积（stride=2）
        self.conv1 = build_conv_layer(
            self.conv_cfg,
            self.in_channels,
            64,
            kernel_size=3,
            stride=2,
            padding=1)
        self.norm1 = build_norm_layer(self.norm_cfg, 64)[1]
        
        # 第二个3x3卷积（stride=1）
        self.conv2 = build_conv_layer(
            self.conv_cfg,
            64,
            64,
            kernel_size=3,
            stride=1,  # 论文中此处为stride=1
            padding=1)
        self.norm2 = build_norm_layer(self.norm_cfg, 64)[1]
        self.relu = nn.ReLU(inplace=True)

    def _make_stages(self):
        """构建HRNet阶段层（需保持原有实现）"""
        # ... 保持原有_stages实现不变 ...

    def forward(self, x):
        """前向传播流程"""
        # Stem层处理
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        
        # 获取HRNet多尺度特征
        hrnet_features = super().forward(x)  # 从父类获取stage特征
        
        # 尺度感知处理
        processed_features = []
        for feat, scale_module in zip(hrnet_features, self.scale_aware_modules):
            processed = scale_module(feat)
            processed_features.append(processed)
            
        # 多分辨率预测
        outputs = []
        for feat, head in zip(processed_features, self.heads):
            heatmap = head['heatmap'](feat)
            tagmap = head['tagmap'](feat)
            
            heatmap = torch.sigmoid(heatmap)  # 按论文要求应用sigmoid
            
            outputs.append({
                'heatmap': heatmap,
                'tagmap': tagmap,
                'feature': feat
            })
            
        return outputs

    def train_step(self, data_batch, optimizer, **kwargs):
        """Training step with multi-resolution supervision."""
        losses = dict()
        
        # Forward
        outputs = self(data_batch['img'])
        
        # Calculate losses at each resolution
        for idx, output in enumerate(outputs):
            scale_weight = 1.0 / (2 ** idx)  # Higher weight for higher resolution
            
            # Heatmap loss
            losses[f'heatmap_loss_{idx}'] = self.heatmap_loss(
                output['heatmap'],
                data_batch[f'heatmap_{idx}']
            ) * scale_weight
            
            # Tag loss
            losses[f'tag_loss_{idx}'] = self.tag_loss(
                output['tagmap'],
                data_batch[f'tagmap_{idx}']
            ) * scale_weight
            
        return losses