_base_ = "../../model_s2anet.py"

_num_classes = 20

_fam_head = dict(
    num_classes=_num_classes,
)
_odm_head = dict(
    num_classes=_num_classes,
)


model = dict(
    fam_head=_fam_head,
    odm_head=_odm_head,
)