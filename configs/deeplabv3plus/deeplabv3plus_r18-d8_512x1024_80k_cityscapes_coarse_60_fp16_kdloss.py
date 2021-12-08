_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/cityscapes_coarse_weight.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

model = dict(
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(depth=18),
    decode_head=dict(
        c1_in_channels=64,
        c1_channels=12,
        in_channels=512,
        channels=128,
        loss_decode=dict(
            type='KDCrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    ),
    auxiliary_head=dict(
        in_channels=256,
        channels=64,
        loss_decode=dict(
            type='KDCrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
    ))

optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)

data_root = 'data/cityscapes/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        img_dir='leftImg8bit/train',
        ann_dir='gtCoarse_60/train',
        weight_dir='gtCoarse_60/train_coarse_60_MultiTeacher_weight_bmscore.csv'))