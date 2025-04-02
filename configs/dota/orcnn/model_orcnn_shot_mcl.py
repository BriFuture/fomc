_base_ = "model_orcnn_base.py"

_num_classes = 15

_bbox_head = dict(
    num_classes=_num_classes,
    type="fomc_RotatedShared2FCBBoxHead",
)

_roi_head = dict(
    type="fomc_OrientedStandardRoIHead",
    bbox_head=_bbox_head,
    cfg = dict(
        contrast = dict(
            decay_steps=[6000, ],
            decay_rate = 0.2,
            contrast_weight_step=100,
            contrast_weight = 0.25,
        ),
        contrast_head = dict(
            type="LinearContrastiveHead",
            dim_in=1024,
            feat_dim=256,
            dim_out=128,
        ),
        contrast_loss = dict(
            type="BankSupConLoss",
            temperature=0.5,
            iou_threshold=0.5,
            bg_iou_threshold=0.25,
            dim=128,
            queue_size=8192,
            reweight_func="trainval",
        ),
    )
)
_train_cfg = dict(
    rcnn=dict(
        sampler=dict(
            type='fs_RRandomSampler',
        ),
    )
)

model = dict(
    train_cfg = _train_cfg,
    roi_head=_roi_head,
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
            "gdet/modules/basic_modules/fsfpn.py",
            "gdet/modules/few_shot/model/fomc_head.py"
        ]
    ),
)
