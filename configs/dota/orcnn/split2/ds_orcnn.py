_base_ = "../ds_orcnn.py"

ALL_CLASSES = ("plane", "ship", "ground-track-field", "harbor", "bridge", "large-vehicle", 
    "small-vehicle", "helicopter", "soccer-ball-field", "swimming-pool",
    "storage-tank", "baseball-diamond", "basketball-court", "tennis-court", "roundabout",
)


NOVEL_CLASSES = ["storage-tank", "baseball-diamond", "basketball-court", "tennis-court", "roundabout",]
BASE_CLASSES  = tuple([av for av in ALL_CLASSES if av not in NOVEL_CLASSES])

dataset = dict(

    train = dict(
        dst_classes = BASE_CLASSES,
        all_classes = ALL_CLASSES,
        novel_classes = NOVEL_CLASSES,
        base_classes = BASE_CLASSES,
    ),
    val = dict(
        dst_classes = BASE_CLASSES,
        all_classes = ALL_CLASSES,
        novel_classes = NOVEL_CLASSES,
        base_classes = BASE_CLASSES,
    ),
    test = dict(
        dst_classes = BASE_CLASSES,
        all_classes = ALL_CLASSES,
        novel_classes = NOVEL_CLASSES,
        base_classes = BASE_CLASSES,
    ),

)