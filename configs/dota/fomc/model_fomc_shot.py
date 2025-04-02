_base_ = "model_fomc_base.py"

_num_classes = 15

_fam_head = dict(
    type="fomc_RotatedRetinaHead",
    num_classes=_num_classes,
)

_contrast_weight = 0.5

_odm_head = dict(
    type="fomc_ODMRefinedHead",
    base_classes=15, # 12 fg
    num_classes=_num_classes,
    cfg=dict(
        cosine_on=False,
        cosine_scale=10,
        omit_fam_cls=False,
        omit_fam_box=False,

        contrast = dict(
            decay_steps=[4000, 8000],
            decay_rate = 0.4,
            contrast_weight_step=100,
            contrast_weight = _contrast_weight,
        ),
        contrast_head = dict(
            type="ConvContrastiveHead",
            dim_in=256,
            dim_out=128,
            feat_dim=128,
        ),
        contrast_loss = dict(
            type="BankSupConLoss",
            temperature=0.5,
            iou_threshold=0.5,
            bg_iou_threshold=0.15,
            dim=128,
            queue_size=8192,
            reweight_func="trainval",
        ),
    ),    
)

model = dict(
    type="FomcDetector",
    few_shot_train = True,
    freeze_backbone=True,
    freeze_neck=True,
    fam_head=_fam_head,
    odm_head=_odm_head,
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