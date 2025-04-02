_base_ = "../ds_r3det.py"

HRSC_CLASSES = ("ship", "aircraft carrier", "warcraft", "merchant ship",
    "Nimitz", "Enterprise", "Arleigh Burke", "WhidbeyIsland",
    "Perry", "Sanantonio", "Ticonderoga", "Kitty Hawk",
    "Kuznetsov", "Abukuma", "Austen", "Tarawa", "Blue Ridge",
    "Container", "OXo|--)", "Car carrier([]==[])",
    "Hovercraft", "yacht", "CntShip(_|.--.--|_]=", "Cruise",
    "submarine", "lute", "Medical", "Car carrier(======|",
    "Ford-class", "Midway-class", "Invincible-class"
)
ALL_CLASSES = HRSC_CLASSES

NOVEL_CLASSES = ["Tarawa", "Austen",
    "CommanderA", "Medical",
    "Car carrier(======|", 
]
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