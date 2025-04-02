_base_ = "ds_dior.py"


dataset = dict(
    train = dict(
        type = 'RotateVocDataset',
        ann_file = ["datasets/DIOR-R/ImageSets/Main/train.txt", ],
        img_dir = "datasets/DIOR-R/",
        img_subdir = "JPEGImages-trainval",
        ann_subdir = "All_Annotations/Oriented Bounding Boxes",
    ),
    val = dict(
        type = 'RotateVocDataset',
        ann_file = "datasets/DIOR-R/ImageSets/Main/val.txt",
        img_dir = "datasets/DIOR-R/",
        ann_subdir = "All_Annotations/Oriented Bounding Boxes",
        img_subdir = "JPEGImages-trainval",
        # img_subdir = "JPEGImages-test",
    ),
    test = dict(
        ann_file = "datasets/DIOR-R/ImageSets/Main/test.txt",
        img_dir = "datasets/DIOR-R/",
        ann_subdir = "All_Annotations/Oriented Bounding Boxes",
        img_subdir = "JPEGImages-test",
    ),
)
