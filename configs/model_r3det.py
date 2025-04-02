_base_ = ["model_fs_base.py"]
### from mmrotate/configs/r3det/r3det_refine_r50_fpn_1x_dota_oc.py
_angle_version = 'oc'
_num_classes=15
_train_cfg = dict(
    s0=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
            iou_calculator=dict(type='RBboxOverlaps2D')),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    stage_loss_weights=[1.0, 1.0]
)
_test_cfg=dict(
    nms_pre=2000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(iou_thr=0.1),
    max_per_img=2000
)

_backbone = dict(
    type='ResNet',
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=1,
    zero_init_residual=False,
    norm_cfg=dict(type='BN', requires_grad=True),
    norm_eval=True,
    style='pytorch',
    init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
)
_neck = dict(
    type='FPN',
    in_channels=[256, 512, 1024, 2048],
    out_channels=256,
    start_level=1,
    add_extra_convs='on_input',
    num_outs=5
)
_bbox_head = dict(
    type='RotatedRetinaHead',
    num_classes=_num_classes,
    in_channels=256,
    stacked_convs=4,
    feat_channels=256,
    anchor_generator=dict(
        type='RotatedAnchorGenerator',
        octave_base_scale=4,
        scales_per_octave=3,
        ratios=[1.0, 0.5, 2.0],
        strides=[8, 16, 32, 64, 128]),
    bbox_coder=dict(
        type='DeltaXYWHAOBBoxCoder',
        angle_range=_angle_version,
        norm_factor=None,
        edge_swap=False,
        proj_xy=False,
        target_means=(.0, .0, .0, .0, .0),
        target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
    loss_cls=dict(
        type='FocalLoss',
        use_sigmoid=True,
        gamma=2.0,
        alpha=0.25,
        loss_weight=1.0),
    loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0),
    train_cfg= _train_cfg['s0'],
    test_cfg= _test_cfg,
)

### for inherit simplicity, change _refine_heads from list into dict
_refine_heads = {
    0: dict(
        type='RotatedRetinaRefineHead',
        num_classes=_num_classes,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        assign_by_circumhbbox=None,
        anchor_generator=dict(
            type='PseudoAnchorGenerator', strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHAOBBoxCoder',
            angle_range=_angle_version,
            norm_factor=None,
            edge_swap=False,
            proj_xy=False,
            target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0),
        train_cfg = dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.6,
                neg_iou_thr=0.5,
                min_pos_iou=0,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='RBboxOverlaps2D')),
            allowed_border=-1,
            pos_weight=-1,
            debug=False
        ),
        test_cfg= _test_cfg,
    ),
    1: dict(
        type='RotatedRetinaRefineHead',
        num_classes=_num_classes,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        assign_by_circumhbbox=None,
        anchor_generator=dict(
            type='PseudoAnchorGenerator', strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHAOBBoxCoder',
            angle_range=_angle_version,
            norm_factor=None,
            edge_swap=False,
            proj_xy=False,
            target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0),
        train_cfg = dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.6,
                min_pos_iou=0,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='RBboxOverlaps2D')),
            allowed_border=-1,
            pos_weight=-1,
            debug=False
        ),
        test_cfg= _test_cfg,
    )
}
model = dict(
    builder = "r3det",
    type='R3Det',
    backbone=_backbone,
    neck=_neck,
    bbox_head=_bbox_head,
    frm_cfgs=[
        dict(in_channels=256, featmap_strides=[8, 16, 32, 64, 128]),
        dict(in_channels=256, featmap_strides=[8, 16, 32, 64, 128])
    ],
    num_refine_stages=2,
    refine_heads=_refine_heads,
    train_cfg=_train_cfg,
    test_cfg=_test_cfg,
    modules = dict(

        s2anet = [
            "mmrotate/models/detectors/r3det.py",
            "mmrotate/models/dense_heads/rotated_retina_head.py",
            "mmdet/models/losses/focal_loss.py",
            "mmdet/models/losses/smooth_l1_loss.py",
            "mmrotate/models/dense_heads/odm_refine_head.py",
            "mmrotate/core/anchor/anchor_generator.py",
            "mmrotate/models/dense_heads/rotated_retina_refine_head.py",
        ]
    ),
)

weights = dict(
    backbone_pretrained = 'torchvision://resnet50',
)