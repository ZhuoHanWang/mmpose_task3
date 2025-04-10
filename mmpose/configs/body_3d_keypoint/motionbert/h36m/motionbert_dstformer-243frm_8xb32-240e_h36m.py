_base_ = ['../../../_base_/default_runtime.py']

vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='Pose3dLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# runtime
train_cfg = dict(max_epochs=240, val_interval=10)

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.01))

# learning policy
param_scheduler = [
    dict(type='ExponentialLR', gamma=0.99, end=120, by_epoch=True)
]

auto_scale_lr = dict(base_batch_size=512)

# hooks
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        save_best='MPJPE',
        rule='less',
        max_keep_ckpts=1),
    logger=dict(type='LoggerHook', interval=20),
)

# codec settings
train_codec = dict(
    type='MotionBERTLabel', num_keypoints=17, concat_vis=True, mode='train')
val_codec = dict(
    type='MotionBERTLabel', num_keypoints=17, concat_vis=True, rootrel=True)

# model settings
model = dict(
    type='PoseLifter',
    backbone=dict(
        type='DSTFormer',
        in_channels=3,
        feat_size=512,
        depth=5,
        num_heads=8,
        mlp_ratio=2,
        seq_len=243,
        att_fuse=True,
    ),
    head=dict(
        type='MotionRegressionHead',
        in_channels=512,
        out_channels=3,
        embedding_size=512,
        loss=dict(type='MPJPEVelocityJointLoss'),
        decoder=val_codec,
    ),
    test_cfg=dict(flip_test=True))

# base dataset settings
dataset_type = 'Human36mDataset'
data_root = 'data/h36m/'

c