_base_ = "model_s2anet_base.py"

_num_classes = 15

_fam_head = dict(
    num_classes=_num_classes,
)
_odm_head = dict(
    num_classes=_num_classes,
)


model = dict(
    fam_head=_fam_head,
    odm_head=_odm_head,
    freeze_rpn = True,
    freeze_backbone = True,
    backbone=dict(
        frozen_stages=4,
    ),
    neck=dict(
        type="FsFPN",
        frozen=[0, 1, 2, 3, 4],
    ),
    modules = dict(
        finetune = [
            "gdet/modules/basic_modules/fsfpn.py"
        ]
    ),
)