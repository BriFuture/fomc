_base_ = "../../model_orcnn.py"

_num_classes = 15

_bbox_head = dict(
    num_classes=_num_classes,
)

_roi_head = dict(
    bbox_head=_bbox_head,
)

model = dict(
    roi_head=_roi_head,
)