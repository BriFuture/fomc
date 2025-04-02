_base_ = "../../model_fomc.py"

_num_classes = 10

_fam_head = dict(
    num_classes=_num_classes,
)


_odm_head = dict(
    num_classes=_num_classes,
)

model = dict(
    # type="FomcDetector",
    fam_head=_fam_head,
    odm_head=_odm_head,

    modules = dict(
        fomc = [
            "gdet/modules/few_shot/model"
        ],
    ),
)