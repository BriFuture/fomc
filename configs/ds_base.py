
_num_classes = 90 + 1

_img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], max_value=255., to_rgb=True)

_train_transforms = [
    dict(type="LoadImageFromFile"),
    dict(type='LoadAnnotations', with_bbox=True, with_id=False),
    dict(type='TF_Resize', img_scale=(512, 512), keep_ratio=True), 
    # dict(type='Pad', size_divisor=64),
    dict(type='TF_Pad', size=(512, 512), pad_val=(0, 0, 0)),
    dict(type="TF_Normalize", **_img_norm_cfg),
    # dict(type="ImageToTensor", keys=['img']), 
    dict(type="DefaultFormatBundle", ), 
    dict(type='TF_Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'text']),
]

_val_transforms = [
    dict(type="LoadImageFromFile"), 
    dict(type='LoadAnnotations', with_bbox=False, with_id=False),
    dict(type='TF_Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='TF_Pad', size=(512, 512), pad_val=(0, 0, 0)),
    dict(type="TF_Normalize", **_img_norm_cfg),
    
    dict(type="DefaultFormatBundle", ), 
    dict(type='TF_Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'text']),
]

_test_transforms = [
    dict(type="LoadImageFromFile"), 
    dict(type='LoadAnnotations', with_bbox=False),
    dict(type="TF_Normalize", **_img_norm_cfg),
    dict(type="ImageToTensor", keys=['img']), 
    dict(type='TF_Collect', keys=['img']),
]

dataset = dict(
    train = dict(
        type = 'CocoDataset',
        ann_file = "datasets/coco/raw_data/annotations/instances_train2017.json",
        img_dir = "datasets/coco/raw_data/train2017",
        num_classes = _num_classes,

        transforms = _train_transforms,
    ),
    val = dict(
        type = 'CocoDataset',
        ann_file = "datasets/coco/raw_data/annotations/instances_val2017.json",
        img_dir = "datasets/coco/raw_data/val2017",
        num_classes = _num_classes,
        transforms = _val_transforms,
    ),
    evaluator = dict(
        type='TestCOCOEvaluator',
        num_classes = _num_classes,
        transforms = _test_transforms,
    ),
    data_augmentation = [],
    data_augmentation_args = dict(
        random_crop = dict(
            scale = 1,
            ratio = 0.5,
        )
    ),
    base_category_indicator = None, ## default None
)


