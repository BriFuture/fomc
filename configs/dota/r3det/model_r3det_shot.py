_base_ = "model_r3det_base.py"

_num_classes = 15

_bbox_head = dict(
    num_classes=_num_classes,
)

_refine_heads = {
    0: dict(
        num_classes=_num_classes,
    ),
    1: dict(
        num_classes=_num_classes,
    ),
}

model = dict(
    few_shot_train = True,
    bbox_head=_bbox_head,
    refine_heads=_refine_heads,
    backbone=dict(
        frozen_stages=4,
    ),
    neck=dict(
        type="FsFPN",
        frozen=[0, 1, 2, 3, 4],
    ),
    cfg = dict(
        consistency_coeff = 1,
        basedet_bonus=0.1,
        basedet_thresh = 0.05,
        use_merged_outputs = True,
    ),
    modules = dict(
        finetune = [
            "gdet/modules/basic_modules/fsfpn.py"
        ],
        fomc = [
            "gdet/modules/few_shot/model"
        ],
    ),
)