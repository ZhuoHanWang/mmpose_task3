_base_ = ['mmpose::_base_/default_runtime.py']
# /root/task2/mmpose/configs/body_2d_keypoint/simcc/coco/simcc_res50_8xb64-210e_coco-256x192.py
# runtime
train_cfg = dict(max_epochs=200, val_interval=1)

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=1e-3,
))

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(type='MultiStepLR', milestones=[170, 200], gamma=0.1, by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

#key 'PCK' return from mmpose.evaluation.metrics.keypoint_2d_metrics.py
default_hooks = dict(checkpoint=dict(save_best='PCK@0.01', rule='greater'))

# codec settings
codec = dict(
    type='SimCCLabel', input_size=(256, 256), sigma=6.0, simcc_split_ratio=2.0)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='ResNet',
        depth=50,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    ),
    head=dict(
        type='SimCCHead',
        in_channels=2048,
        out_channels=17,
        input_size=codec['input_size'],
        in_featuremap_size=tuple([s // 32 for s in codec['input_size']]),
        simcc_split_ratio=codec['simcc_split_ratio'],
        loss=dict(type='KLDiscretLoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(flip_test=True))

# base dataset settings
dataset_type = 'SpineDataset'
data_mode = 'topdown'
data_root = '/pangyan/wzh/mmpose_task/task3/data'   # 记得改数据根路径

# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/sample/spine_keypoints_xray_1_v2_train.json',
        data_prefix=dict(img='images/'),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/sample/spine_keypoints_xray_1_v2_val.json',
        bbox_file=None,
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# evaluators
#mmpose.evaluation.metrics.keypoint_2d_metrics.py
val_evaluator = dict(
    # type='PCKAccuracy',
    # thr=0.05,
    type='SpineAccuracy',
    thr_list=[0.01, 0.03, 0.05, 0.1, 0.15],
)
test_evaluator = val_evaluator

visualizer = dict(
    cobb_path = "/root/task3/mmpose/",
    draw_bbox=False,
    vis_backends=[
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    ],
    radius=4
)