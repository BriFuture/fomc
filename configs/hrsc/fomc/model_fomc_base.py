_base_ = "../../model_fomc.py"

_num_classes = 15
_train_cfg = dict(
    assigner=dict(
        min_pos_iou=0,
    ),
)
_fam_head = dict(
    num_classes=_num_classes,
    train_cfg = _train_cfg,
)


_odm_head = dict(
    num_classes=_num_classes,
    train_cfg = _train_cfg,

)

model = dict(
    # type="FomcDetector",
    fam_head=_fam_head,
    odm_head=_odm_head,
    modules = dict(
        fomc = [
            "gdet/modules/few_shot/model",
        ],
    ),
)