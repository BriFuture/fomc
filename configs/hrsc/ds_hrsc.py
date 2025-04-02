_base_ = "../ds_base.py"

_angle_version = "oc"

_img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], max_value=255., to_rgb=True)


ALL_CLASSES = ["ship", "aircraft carrier", "warcraft", "merchant ship", 
    "Nimitz", "Enterprise",
    "Arleigh Burke", "WhidbeyIsland", "Perry", "Sanantonio",  "Ticonderoga",
    "Container", "Car-carrierA", "Hovercraft",  "ContainerA",
    "submarine",

    "Tarawa", "Austen", "CommanderA", "Medical", "Car-carrierB",
]
HRSC_CLASSES = ("ship", "aircraft carrier", "warcraft", "merchant ship",
    "Nimitz", "Enterprise", "Arleigh Burke", "WhidbeyIsland",
    "Perry", "Sanantonio", "Ticonderoga", "Kitty Hawk",
    "Kuznetsov", "Abukuma", "Austen", "Tarawa", "Blue Ridge",
    "Container", "OXo|--)", "Car carrier([]==[])",
    "Hovercraft", "yacht", "CntShip(_|.--.--|_]=", "Cruise",
    "submarine", "lute", "Medical", "Car carrier(======|",
    "Ford-class", "Midway-class", "Invincible-class"
)
HRSC_CLASSES_ID = ("01", "02", "03", "04", "05", "06", "07", "08", "09",
    "10", "11", "12", "13", "14", "15", "16", "17", "18",
    "19", "20", "22", "24", "25", "26", "27", "28", "29",
    "30", "31", "32", "33"
)
HRSC_ID_NAME_MAP = {
    "01": "ship", "02": "aircraft carrier", "03": "warcraft", "04": "merchant ship", "05": "Nimitz", 
    "06": "Enterprise", "07": "Arleigh Burke", "08": "WhidbeyIsland", "09": "Perry", "10": "Sanantonio", 
    "11": "Ticonderoga", "12": "Kitty Hawk", "13": "Kuznetsov", "14": "Abukuma", "15": "Austen", "16": "Tarawa", 
    "17": "Blue Ridge", "18": "Container", "19": "OXo|--)", "20": "Car carrier([]==[])", 
    "22": "Hovercraft", "24": "yacht", "25": "CntShip(_|.--.--|_]=", "26": "Cruise", 
    "27": "submarine", "28": "lute", "29": "Medical", "30": "Car carrier(======|", "31": "Ford-class", 
    "32": "Midway-class", "33": "Invincible-class"
}
CLASS_INHERITANCE = {
    "01": "01",
    "02": "01", "03": "01", "04": "01", "27": "01",

    "05": "02", "06": "02", "12": "02", "13": "02", 
    "16": "02", "31": "02", "32": "02", "33": "02",

    "07": "03", "08": "03", "09": "03", "10": "03",
    "11": "03", "14": "03", "15": "03", "17": "03", 
    "19": "03", "28": "03", "29": "03", 
    
    "18": "04", "20": "04", "22": "04", "24": "04",
    "25": "04", "26": "04", "30": "04", 
}
ALL_CLASSES = HRSC_CLASSES

_num_classes = len(ALL_CLASSES) + 1

NOVEL_CLASSES = [ "Tarawa", "Austen",
    "CommanderA", "Medical",
    "Car-carrierB",
]
### 与 HRSC_ID_NAME_MAP 一致
CLASS_IDSTR_NAME = {
    # level 1
    "01": "ship",   
    # level 2
    "02": "aircraft-carrier", "03": "warcraft",      "04": "merchant-ship",
    # level 3
    ## 航空母舰
    "05": "Nimitz", 
    "06": "Enterprise",       
    "12": "Kitty-Hawk", 
    "13": "Kuznetsov",  
    "16": "Tarawa", 
    "31": "Ford-class", 
    "32": "Midway-class", 
    "33": "Invincible-class",
    ## 战舰
    "07": "Arleigh-Burke",   
    "08": "WhidbeyIsland",   
    "09": "Perry",  
    "10": "Sanantonio",      
    "11": "Ticonderoga",   
    "14": "Abukuma",      
    "15": "Austen",        
    "17": "Blue-Ridge",  # 
    "19": "CommanderA",  # "OXo|--)"          
    "28": "lute",
    "29": "Medical",       
    ## 商船
    "18": "Container",    
    "20": "Car-carrierA",  # Car carrier([]==[])
    "22": "Hovercraft", 
    "24": "yacht",        
    "25": "ContainerA",   #  CntShip(_|.--.--|_]=    
    "26": "Cruise", 
    "30": "Car-carrierB", # Car carrier(======|
    
    ## 潜艇
    "27": "submarine",   
}

TRAIN_CLASSES = tuple([av for av in ALL_CLASSES if av not in NOVEL_CLASSES])
VAL_CLASSES = ALL_CLASSES[:]

_base_category_indicator = tuple([True] + [av in TRAIN_CLASSES for av in ALL_CLASSES])
_train_transform = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True, with_id=True),
    dict(type="TF_RResize", img_scale=[(800, 800),], keep_ratio=True), 
    dict(type="TF_RRandomFlip", flip_ratio=0.5),
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
            dict(type="TF_Collect", keys=["img"])
        ]
    )
]
_bg_offset = 0

_data_root = "datasets/HRSC2016/FullDataSet/"

dataset = dict(
    train = dict(
        type = "HRSCDataset",
        ann_file = [_data_root + "ImageSets/trainval.txt", ],
        img_dir = _data_root,
        ann_subdir= "Annotations/",
        img_subdir= "AllImages/",
        img_prefix= "AllImages/",

        num_classes = _num_classes,
        dst_classes = ALL_CLASSES,
        classes_id2name = HRSC_ID_NAME_MAP,
        transforms = _train_transform,
        bg_offset = _bg_offset,
        classwise=True,
        version=_angle_version,
    ),
    val = dict(
        type = "HRSCDataset",
        ann_file = _data_root + "ImageSets/test.txt",
        img_dir = _data_root,
        ann_subdir= "Annotations/",
        img_subdir= "AllImages/",
        img_prefix= "AllImages/",
        num_classes = _num_classes,
        dst_classes = ALL_CLASSES,
        classes_id2name = HRSC_ID_NAME_MAP,
        transforms = _test_transform,
        bg_offset = _bg_offset,
        classwise=True,
        version=_angle_version,
    ),
    test = dict(
        type = "HRSCDataset",
        ann_file = _data_root + "ImageSets/test.txt",
        img_dir = _data_root,
        ann_subdir= "Annotations/",
        img_subdir= "AllImages/",
        img_prefix= "AllImages/",
        dst_classes = ALL_CLASSES,
        classes_id2name = HRSC_ID_NAME_MAP,
        transforms = _test_transform,
        bg_offset = _bg_offset,
        classwise=True,
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
            "gdet/datasets/dataset/hrsc.py",
            "gdet/datasets/transforms/rotate_transforms.py",
            "gdet/datasets/evaluator/dota.py",
            "gdet/datasets/dataset/rotated_voc_dataset.py",
            "gdet/datasets/evaluator/fs_rotate_voc_evaluator.py",
        ]
    )
)
