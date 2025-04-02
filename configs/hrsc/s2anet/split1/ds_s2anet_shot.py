_base_ = "ds_s2anet.py"

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