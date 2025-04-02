_base_ = "model_orcnn_base.py"

_num_classes = 15

_bbox_head = dict(
    num_classes=_num_classes,
)

_roi_head = dict(
    bbox_head=_bbox_head,
)

model = dict(
    roi_head=_roi_head,
    few_shot_train=False,
    freeze_backbone=True,
    freeze_neck=True,
    freeze_rpn=False,
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