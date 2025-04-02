_base_ = ["ds_s2anet.py", "../rt_dior.py", "../../model_s2anet.py"]

experiment = dict(
    work_dir="checkpoints/dior_r/s2anet/split1/base",
)

train = dict(
    solver = dict(
        learning_rate = 0.01,
        decay_epochs = [ 8, 11],
        total_epochs = 12,
    ),
)
ALL_CLASSES = ["airport", "bridge", "dam", "harbor", "overpass", "ship", "vehicle", 
    "basketballcourt", 
    "chimney",  "Expressway-Service-area", "Expressway-toll-station", "golffield", 
    "groundtrackfield",   "stadium", 
    "storagetank", 
    "airplane", "baseballfield", "tenniscourt", "trainstation", "windmill", 
]

TRAIN_CLASSES = ["airport", "bridge", "dam", "harbor", "overpass", "ship", "vehicle", ]
NOVEL_CLASSES = tuple([av for av in ALL_CLASSES if av not in TRAIN_CLASSES])
VAL_CLASSES = TRAIN_CLASSES[:]

dataset = dict(

    train = dict(
        dst_classes = TRAIN_CLASSES,
        all_classes = ALL_CLASSES,
    ),
    val = dict(
        dst_classes = VAL_CLASSES,
        all_classes = ALL_CLASSES,
    ),
    test = dict(
        dst_classes = VAL_CLASSES,
        all_classes = ALL_CLASSES,
    ),
)