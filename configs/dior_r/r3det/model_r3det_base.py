_base_ = "../../model_r3det.py"

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
    bbox_head=_bbox_head,
    refine_heads=_refine_heads,

    modules = dict(
        fomc = [
            "gdet/modules/few_shot/model"
        ],
    ),
)