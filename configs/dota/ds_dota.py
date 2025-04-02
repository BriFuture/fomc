_base_ = "../ds_base.py"
_angle_version = 'oc'

ALL_CLASSES = ("plane", "baseball-diamond", "bridge", "ground-track-field",
            "small-vehicle", "large-vehicle", "ship", "tennis-court",
            "basketball-court", "storage-tank", "soccer-ball-field",
            "roundabout", "harbor", "swimming-pool", "helicopter")
_data_root = "datasets/DOTA_v1/"
dataset = dict(
    train = dict(
        type = "DOTADataset",
        dst_classes = ALL_CLASSES,
        img_dir = _data_root+"train_ss/images",
        ann_folder = _data_root+"train_ss/annfiles",
        version=_angle_version,
        difficulty=100,
    ),
    val = dict(
        type = "DOTADataset",
        dst_classes = ALL_CLASSES,
        img_dir = _data_root+"val_ss/images",
        ann_folder = _data_root+"val_ss/annfiles",
        version=_angle_version,
    ),
    test = dict(
        type = "DOTADataset",
        dst_classes = ALL_CLASSES,
        img_dir = _data_root+"test/images",
        ann_folder = _data_root+"test/images",
        version=_angle_version,
    ),
    evaluator = dict(
        # num_classes = _num_classes,
        type="FS_RotatedBoxEvaluator",
    ),
    PALETTE = [
        (165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
        (138, 43, 226), (255, 128, 0), (255, 0, 255), (0, 255, 255),
        (255, 193, 193), (0, 51, 153), (255, 250, 205), (0, 139, 139),
        (255, 255, 0), (147, 116, 116), (0, 0, 255)
    ],
    modules = dict(
        dota = [
            "gdet/datasets/dataset/dota.py",
            "gdet/datasets/transforms/rotate_transforms.py",
            "gdet/datasets/evaluator/dota.py",
            "gdet/datasets/evaluator/fs_rotate_voc_evaluator.py",
        ]
    )
)
