_base_ = "../ds_r3det.py"

ALL_CLASSES = ["airport", "basketballcourt", "bridge", 
    "chimney", "dam", "Expressway-Service-area", "Expressway-toll-station", "golffield", 
    "groundtrackfield", "harbor", "overpass", "ship",  "stadium", 
    "storagetank",  "vehicle",   
    "airplane", "baseballfield", "tenniscourt", "trainstation", "windmill", 
]

NOVEL_CLASSES = ["airplane", 'baseballfield', 'tenniscourt', "trainstation", "windmill",]
TRAIN_CLASSES = tuple([av for av in ALL_CLASSES if av not in NOVEL_CLASSES])
VAL_CLASSES = TRAIN_CLASSES[:]

dataset = dict(

    train = dict(
        all_classes = ALL_CLASSES,
    ),
    val = dict(
        all_classes = ALL_CLASSES,
    ),
    test = dict(
        all_classes = ALL_CLASSES,
    ),
    evaluator_val = dict(
        type="FS_RotatedBoxEvaluator",
    ),
)