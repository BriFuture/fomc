_base_ = "../ds_s2anet.py"

ALL_CLASSES = ("plane", "ship", "storage-tank", "baseball-diamond", "tennis-court", 
    "basketball-court", "ground-track-field", "harbor", "large-vehicle",  "roundabout", 
    "bridge", "small-vehicle", "helicopter", "soccer-ball-field", "swimming-pool"
)

NOVEL_CLASSES = ["bridge", "small-vehicle", "helicopter", "soccer-ball-field", "swimming-pool"]
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