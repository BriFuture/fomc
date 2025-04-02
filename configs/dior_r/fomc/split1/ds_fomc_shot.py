_base_ = "ds_fomc.py"

ALL_CLASSES = ["airport", "basketballcourt", "bridge", 
    "chimney", "dam", "Expressway-Service-area", "Expressway-toll-station", "golffield", 
    "groundtrackfield", "harbor", "overpass", "ship",  "stadium", 
    "storagetank",  "vehicle",   
    "airplane", "baseballfield", "tenniscourt", "trainstation", "windmill", 
]

dataset = dict(
    train = dict(
        dst_classes = ALL_CLASSES,
    ),
    val = dict(
        dst_classes = ALL_CLASSES,
    ),
    test = dict(
        dst_classes = ALL_CLASSES,
    ),
)