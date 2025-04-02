_base_ = "../ds_base.py"

_num_classes = 20 + 1
_angle_version = 'oc'

_img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], max_value=255., to_rgb=True)


ALL_CLASSES = ["airport", "basketballcourt", "bridge", 
    "chimney", "dam", "Expressway-Service-area", "Expressway-toll-station", "golffield", 
    "groundtrackfield", "harbor", "overpass", "ship",  "stadium", 
    "storagetank",  "vehicle",   
    "airplane", "baseballfield", "tenniscourt", "trainstation", "windmill", 
]

NOVEL_CLASSES = ["airplane", 'baseballfield', 'tenniscourt', "trainstation", "windmill",]
TRAIN_CLASSES = tuple([av for av in ALL_CLASSES if av not in NOVEL_CLASSES])
VAL_CLASSES = ALL_CLASSES[:]
_base_category_indicator = tuple([True] + [av in TRAIN_CLASSES for av in ALL_CLASSES])
_train_transform = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True, with_id=True),
    dict(type="TF_RResize", img_scale=[(800, 800), ], keep_ratio=True), 
    dict(type='TF_RRandomFlip', flip_ratio=0.5),
    dict(type="TF_Pad", size_divisor=32),
    dict(type="TF_Normalize", **_img_norm_cfg),
    dict(type="DefaultFormatBundle", ), 
    dict(type="TF_Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]

_test_transform = [
    dict(type="LoadImageFromFile"),
    dict(
        type="TF_MultiScaleFlipAug",
        img_scale=(800, 800),
        single_scale = True,
        flip=False,
        transforms=[
            dict(type="LoadAnnotations", with_bbox=False, with_id=True),
            dict(type="TF_RResize"),
            dict(type="TF_Pad", size_divisor=32),
            dict(type="TF_Normalize", **_img_norm_cfg),
            dict(type="DefaultFormatBundle"),
            dict(type="TF_Collect", keys=["img", "ids"])
        ]
    )
]
_bg_offset = 0
dataset = dict(
    train = dict(
        type = 'RotateVocDataset',
        ann_file = ["datasets/DIOR-R/ImageSets/Main/train.txt", ],
        img_dir = "datasets/DIOR-R/",
        img_subdir = "JPEGImages-trainval",
        ann_subdir = "Annotations",
        num_classes = _num_classes,
        dst_classes = ALL_CLASSES,
        transforms = _train_transform,
        bg_offset = _bg_offset,
        version=_angle_version,
    ),
    val = dict(
        type = 'RotateVocDataset',
        ann_file = "datasets/DIOR-R/ImageSets/Main/val.txt",
        img_dir = "datasets/DIOR-R/",
        ann_subdir = "Annotations",
        # img_subdir = "JPEGImages-test",
        num_classes = _num_classes,
        dst_classes = ALL_CLASSES,
        transforms = _test_transform,
        bg_offset = _bg_offset,
        version=_angle_version,
    ),
    test = dict(
        ann_file = "datasets/DIOR-R/ImageSets/Main/test.txt",
        img_dir = "datasets/DIOR-R/",
        ann_subdir = "Annotations",
        img_subdir = "JPEGImages-test",
        dst_classes = ALL_CLASSES,
        transforms = _test_transform,
        bg_offset = _bg_offset,
        version=_angle_version,
    ),
    evaluator = dict(
        num_classes = _num_classes,
        type="FS_RotatedBoxEvaluator",
    ),
    base_category_indicator = _base_category_indicator,
    modules = dict(
        dota = [
            "gdet/datasets/dataset/dota.py",
            "gdet/datasets/transforms/rotate_transforms.py",
            "gdet/datasets/evaluator/dota.py",
            "gdet/datasets/dataset/rotated_voc_dataset.py",
            "gdet/datasets/evaluator/fs_rotate_voc_evaluator.py",
        ]
    )
)
