_base_ = "ds_dota.py"

_data_root = "datasets/DOTA_v1/"
dataset = dict(
    train = dict(
        img_dir = _data_root+"trainA/images",
        ann_folder = _data_root+"trainA/labelTxt",
        difficulty=100,
    ),
    val = dict(
        img_dir = _data_root+"valA/images",
        ann_folder = _data_root+"valA/annfiles",
    ),
    test = dict(
        img_dir = _data_root+"testA/images",
        ann_folder = _data_root+"testA/images",
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
