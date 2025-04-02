_base_ = "ds_r3det.py"


dataset = dict(
    train = dict(
        dst_class_from_all = True,
    ),
    val = dict(
        dst_class_from_all = True,
    ),
    test = dict(
        dst_class_from_all = True,
    ),
)