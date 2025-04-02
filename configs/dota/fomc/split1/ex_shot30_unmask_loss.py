_base_ = ["ds_fomc_shot.py", "rt_shot.py", "../model_fomc_shot.py"]

_split=1
_shot=30
_seed=5

experiment = dict(
    work_dir=f"checkpoints/dota/fomc/split{_split}/shot{_shot}_unmask_loss",
)

_data_root = f"datasets/DOTA_v1/train_ss/split/mask_seed{_seed}_shot{_shot}/"

_angle_version = 'le135'

_img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], max_value=255., to_rgb=True)
_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_ignore_bbox=True),
    dict(type='TF_RResize', img_scale=(1024, 1024)),
    dict(
        type='TF_RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=_angle_version),
    dict(type='TF_Pad', size_divisor=32),
    dict(type='TF_Normalize', **_img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='TF_Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore'])
]
dataset = dict(

    train = dict(
        shot = _shot,
        ann_file= _data_root + "ImageSets/Main/train.txt",
        img_dir = _data_root + "images",
        ann_folder = _data_root+"annfiles",
        transforms=_train_pipeline,
        difficulty=9,
    ),
)

train = dict(
    log_interval = 20,
    solver = dict(
        decay_epochs = [180,],
        total_epochs = 200,
        learning_rate = 0.0025,
    ),
    eval_checkpoints=5,
)

_fam_head = dict(
    train_cfg = dict(
        assigner=dict(
            ignore_iof_thr=0.1,
        ),
    ),
)
_odm_head = dict(
    type = "fomc_ODMRefinedHeadv2",
    cfg=dict(
        ignore_unused_object_loss = True,
        contrast = dict(
            decay_steps=[4000, ],
            decay_rate = 0.3,
            contrast_weight_step=100,
            contrast_weight = 0.0,
        ),
    ),
    train_cfg = dict(
        assigner=dict(
            ignore_iof_thr=0.1,
        ),
    ),
)

model = dict(
    fam_head = _fam_head,
    odm_head = _odm_head,
)