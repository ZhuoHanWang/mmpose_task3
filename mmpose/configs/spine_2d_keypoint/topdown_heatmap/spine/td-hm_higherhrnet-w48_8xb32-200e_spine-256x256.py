_base_ = ['../../../_base_/default_runtime.py']
# 需要自底向上的头部，不是自顶向下的模型
# runtime
train_cfg = dict(max_epochs=200, val_interval=1)

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=5e-4,
))

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=210,
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

#key 'PCK' return from mmpose.evaluation.metrics.keypoint_2d_metrics.py
default_hooks = dict(checkpoint=dict(save_best='PCK@0.01', rule='greater'))

# codec settings
codec = dict(
    type='MSRAHeatmap',
    input_size=(256, 256),
    heatmap_size=(64, 64),  # 统一分辨率
    sigma=2)

# model settings
model = dict(
    type='BottomupPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='HigherHRNet',
        in_channels=3,
        num_joints=17,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4,),
                num_channels=(64,)),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(48, 96)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(48, 96, 192)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(48, 96, 192, 384),
                multiscale_output=True),
            upsample=dict(
                mode='bilinear',
                align_corners=False)
        ),
        init_cfg=dict(
            type='Pretrained', 
            checkpoint='https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet48_coco_512x512-60fedcbc_20200712.pth'
        )),
    head=dict(
        type='HigherHRNetHead',
        in_channels=[48, 96, 192, 384], 
        num_joints=17,
        tag_per_joint=True,
        deconv_out_channels=[48, 48],  # 修改为相同的通道数
        deconv_kernel_sizes=[4, 4],
        final_conv_kernel=1,
        loss=dict(
            type='MultiLossFactory',
            num_joints=17,
            num_stages=2,
            with_heatmaps_loss=[True, True],
            with_ae_loss=[True, True],
            heatmaps_loss=dict(
                type='MultiResolutionMSELoss',
                use_target_weight=True),
            ae_loss=dict(
                type='MultiscaleAELoss',
                use_target_weight=True)),
        decoder=codec),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True,
        use_nms=True,
        nms_kernel=5,
        nms_padding=2,
        tag_per_joint=True,
        detection_threshold=0.1,
        tag_threshold=1,
        use_detection_val=True,
        ignore_too_much=False))

train_cfg = dict(
    num_scales=2,
    scale_aware_sigma=False)

test_cfg = dict(
    flip_test=True,
    post_process='default',
    shift_heatmap=True,
    modulate_kernel=11)

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
    cobb_path = "/root/task3/mmpose",
    draw_bbox=False,
    vis_backends=[
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    ],
    radius=4
)
