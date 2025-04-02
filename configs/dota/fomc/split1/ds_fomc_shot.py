_base_ = "ds_fomc.py"


dataset = dict(
    train = dict(
        dst_class_from_all = True,
        difficulty=100,
    ),
    val = dict(
        dst_class_from_all = True,
    ),
    test = dict(
        dst_class_from_all = True,
    ),
)