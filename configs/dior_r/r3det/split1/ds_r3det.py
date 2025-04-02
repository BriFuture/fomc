_base_ = "../ds_r3det.py"

ALL_CLASSES = ["airport", "basketballcourt", "bridge", 
    "chimney", "dam", "Expressway-Service-area", "Expressway-toll-station", "golffield", 
    "groundtrackfield", "harbor", "overpass", "ship",  "stadium", 
    "storagetank",  "vehicle",   
    "airplane", "baseballfield", "tenniscourt", "trainstation", "windmill", 
]

NOVEL_CLASSES = ["airplane", 'baseballfield', 'tenniscourt', "trainstation", "windmill",]
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