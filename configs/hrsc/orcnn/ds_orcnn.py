_base_ = "../ds_orcnn.py"

HRSC_CLASSES = ("ship", "aircraft carrier", "warcraft", "merchant ship",
    "Nimitz", "Enterprise", "Arleigh Burke", "WhidbeyIsland",
    "Perry", "Sanantonio", "Ticonderoga", "Kitty Hawk",
    "Kuznetsov", "Abukuma", "Austen", "Tarawa", "Blue Ridge",
    "Container", "OXo|--)", "Car carrier([]==[])",
    "Hovercraft", "yacht", "CntShip(_|.--.--|_]=", "Cruise",
    "submarine", "lute", "Medical", "Car carrier(======|",
    "Ford-class", "Midway-class", "Invincible-class"
)
# ALL_CLASSES = HRSC_CLASSES
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

NOVEL_CLASSES = ["Tarawa", "Austen",
    "CommanderA", "Medical",
    "Car carrier(======|", 
    ]

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

)