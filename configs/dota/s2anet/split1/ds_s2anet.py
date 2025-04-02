_base_ = "../ds_s2anet.py"

ALL_CLASSES = ( "storage-tank", "baseball-diamond", "tennis-court", 
            "basketball-court", "bridge",  
            "small-vehicle", "helicopter", "roundabout", "soccer-ball-field", "swimming-pool",
            "plane", "large-vehicle", "ship", "ground-track-field", "harbor"
)

NOVEL_CLASSES = ["plane", "large-vehicle", "ship", "ground-track-field", "harbor"]
TRAIN_CLASSES = tuple([av for av in ALL_CLASSES if av not in NOVEL_CLASSES])
VAL_CLASSES = TRAIN_CLASSES[:]
BASE_CLASSES  = tuple([av for av in ALL_CLASSES if av not in NOVEL_CLASSES])

dataset = dict(

    train = dict(
        dst_classes = TRAIN_CLASSES,
        all_classes = ALL_CLASSES,
        novel_classes = NOVEL_CLASSES,
        base_classes = BASE_CLASSES,
    ),
    val = dict(
        dst_classes = VAL_CLASSES,
        all_classes = ALL_CLASSES,
        novel_classes = NOVEL_CLASSES,
        base_classes = BASE_CLASSES,
    ),
    test = dict(
        dst_classes = VAL_CLASSES,
        all_classes = ALL_CLASSES,
        novel_classes = NOVEL_CLASSES,
        base_classes = BASE_CLASSES,
    ),
)