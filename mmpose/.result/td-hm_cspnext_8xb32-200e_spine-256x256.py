auto_scale_lr = dict(base_batch_size=512)
backend_args = dict(backend='local')
base_lr = 0.0005
codec = dict(
    heatmap_size=(
        64,
        64,
    ),
    input_size=(
        256,
        256,
    ),
    sigma=2,
    type='MSRAHeatmap')
custom_hooks = [
    dict(
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        priority=49,
        type='EMAHook',
        update_buffers=True),
]
data_mode = 'topdown'
data_root = '/root/task2/dataset'
dataset_type = 'SpineDataset'
default_hooks = dict(
    badcase=dict(
        badcase_thr=5,
        enable=False,
        metric_type='loss',
        out_dir='badcase',
        type='BadCaseAnalysisHook'),
    checkpoint=dict(
        interval=10,
        rule='greater',
        save_best='PCK@0.01',
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(
        enable=True,
        interval=1,
        out_dir='visualization',
        show=False,
        type='PoseVisualizationHook'))
default_scope = 'mmpose'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = '/root/task2/mmpose/work_dirs//td-hm_cspnext_8xb32-200e_spine-256x256/best_PCK@0.01_epoch_166.pth'
log_level = 'INFO'
log_processor = dict(
    by_epoch=True, num_digits=6, type='LogProcessor', window_size=50)
max_epochs = 200
model = dict(
    backbone=dict(
        _scope_='mmdet',
        act_cfg=dict(type='SiLU'),
        arch='P5',
        channel_attention=True,
        deepen_factor=0.67,
        expand_ratio=0.5,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-m_8xb256-rsb-a1-600e_in1k-ecb3bbd9.pth',
            prefix='backbone.',
            type='Pretrained'),
        norm_cfg=dict(type='SyncBN'),
        out_indices=(4, ),
        type='CSPNeXt',
        widen_factor=0.75),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='PoseDataPreprocessor'),
    head=dict(
        decoder=dict(
            heatmap_size=(
                64,
                64,
            ),
            input_size=(
                256,
                256,
            ),
            sigma=2,
            type='MSRAHeatmap'),
        in_channels=768,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        out_channels=17,
        type='HeatmapHead'),
    test_cfg=dict(flip_mode='heatmap', flip_test=False, shift_heatmap=False),
    type='TopdownPoseEstimator')
optim_wrapper = dict(
    optimizer=dict(lr=0.0005, type='Adam'), type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=210,
        gamma=0.1,
        milestones=[
            170,
            200,
        ],
        type='MultiStepLR'),
]
randomness = dict(seed=21)
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/sample/spine_keypoints_rgb_1_v2_val.json',
        data_mode='topdown',
        data_prefix=dict(img='images/'),
        data_root='/root/task2/dataset',
        pipeline=[
            dict(backend_args=dict(backend='local'), type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(input_size=(
                256,
                256,
            ), type='TopdownAffine', use_udp=True),
            dict(type='PackPoseInputs'),
        ],
        test_mode=True,
        type='SpineDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    thr_list=[
        0.01,
        0.03,
        0.05,
        0.1,
        0.15,
    ], type='SpineAccuracy')
train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=1)
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file='annotations/sample/spine_keypoints_rgb_1_v2_train.json',
        data_mode='topdown',
        data_prefix=dict(img='images/'),
        data_root='/root/task2/dataset',
        pipeline=[
            dict(backend_args=dict(backend='local'), type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(direction='horizontal', type='RandomFlip'),
            dict(type='RandomHalfBody'),
            dict(type='RandomBBoxTransform'),
            dict(input_size=(
                256,
                256,
            ), type='TopdownAffine', use_udp=True),
            dict(
                encoder=dict(
                    heatmap_size=(
                        64,
                        64,
                    ),
                    input_size=(
                        256,
                        256,
                    ),
                    sigma=2,
                    type='MSRAHeatmap'),
                type='GenerateTarget'),
            dict(type='PackPoseInputs'),
        ],
        type='SpineDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=dict(backend='local'), type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(direction='horizontal', type='RandomFlip'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(input_size=(
        256,
        256,
    ), type='TopdownAffine', use_udp=True),
    dict(
        encoder=dict(
            heatmap_size=(
                64,
                64,
            ),
            input_size=(
                256,
                256,
            ),
            sigma=2,
            type='MSRAHeatmap'),
        type='GenerateTarget'),
    dict(type='PackPoseInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/sample/spine_keypoints_rgb_1_v2_val.json',
        data_mode='topdown',
        data_prefix=dict(img='images/'),
        data_root='/root/task2/dataset',
        pipeline=[
            dict(backend_args=dict(backend='local'), type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(input_size=(
                256,
                256,
            ), type='TopdownAffine', use_udp=True),
            dict(type='PackPoseInputs'),
        ],
        test_mode=True,
        type='SpineDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    thr_list=[
        0.01,
        0.03,
        0.05,
        0.1,
        0.15,
    ], type='SpineAccuracy')
val_pipeline = [
    dict(backend_args=dict(backend='local'), type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(input_size=(
        256,
        256,
    ), type='TopdownAffine', use_udp=True),
    dict(type='PackPoseInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    draw_bbox=False,
    name='visualizer',
    radius=4,
    type='PoseLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ])
work_dir = '.result'
