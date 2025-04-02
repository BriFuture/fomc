_base_ = "model_s2anet.py"

_fam_head = dict(
    loss_cls = dict(
        use_sigmoid=False,
    ),
    init_cfg = dict(type='Normal', layer='Conv2d', std=0.01,),
)

_odm_head = dict(
    type="ODM_Head",
    loss_cls = dict(
        use_sigmoid=False,
    ),
    init_cfg = dict(type='Normal', layer='Conv2d', std=0.01,),
    cfg = dict(
        use_mlp_cls = False,
    ),
)

model = dict(
    odm_head=_odm_head,
    fam_head=_fam_head,
)