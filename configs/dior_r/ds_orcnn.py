_base_ = "ds_dior.py"
_angle_version = 'le90'
_img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], max_value=255., to_rgb=True)

_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='TF_RResize', img_scale=(800, 800)),
    dict(
        type='TF_RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=_angle_version),
    dict(type='TF_Pad', size_divisor=32),
    dict(type='TF_Normalize', **_img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='TF_Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='TF_MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type="LoadAnnotations", with_bbox=False, with_id=True),
            dict(type='TF_RResize'),
            dict(type='TF_Pad', size_divisor=32),
            dict(type='TF_Normalize', **_img_norm_cfg),
            dict(type='DefaultFormatBundle'),
            dict(type='TF_Collect', keys=['img', "ids"])
        ]
    )
]
dataset = dict(
    train=dict(transforms=_train_pipeline, version=_angle_version),
    val=dict(transforms=_test_pipeline, version=_angle_version),
    test=dict(transforms=_test_pipeline, version=_angle_version),
)
