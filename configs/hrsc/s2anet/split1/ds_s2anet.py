_base_ = "../ds_s2anet.py"

ALL_CLASSES = ("ship", 
    # "aircraft carrier", 
    "warcraft", 
    # "merchant ship",  
    "Nimitz", "Enterprise", "Arleigh Burke", "WhidbeyIsland",
    "Perry", "Sanantonio", "Ticonderoga", 
    # "Kitty Hawk",
    # "Kuznetsov", 
    # "Abukuma",  "Blue Ridge",
    "Container",  "Car carrier([]==[])",
    "Hovercraft", 
    # "yacht", 
    "CntShip(_|.--.--|_]=", 
    # "Cruise",
    "submarine", 
    # "lute", 
    # "Ford-class", 
    "Midway-class", 
    # "Invincible-class",
    "Tarawa", "Austen", "OXo|--)", "Medical", "Car carrier(======|",
)

NOVEL_CLASSES = ["Tarawa", "Austen", "OXo|--)", "Medical", "Car carrier(======|",]
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