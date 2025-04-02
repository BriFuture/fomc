_base_ = ["model_fs_base.py"]
### from mmrotate/configs/s2anet/s2anet_r50_fpn_1x_dota_1e135.py
_angle_version = 'le135'
_num_classes = 15

_fam_head = dict(
    type='RotatedRetinaHead',
    num_classes=_num_classes,
    in_channels=256,
    stacked_convs=2,
    feat_channels=256,
    assign_by_circumhbbox=None,
    anchor_generator=dict(
        type='RotatedAnchorGenerator',
        scales=[4],
        ratios=[1.0],
        strides=[8, 16, 32, 64, 128]),
    bbox_coder=dict(
        type='DeltaXYWHAOBBoxCoder',
        angle_range=_angle_version,
        norm_factor=1,
        edge_swap=False,
        proj_xy=True,
        target_means=(.0, .0, .0, .0, .0),
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
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
            iou_calculator=dict(type='RBboxOverlaps2D')),
        allowed_border=-1,
        pos_weight=-1,
        debug=False
    ),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=2000
    )
)
_odm_head = dict(
    type='ODMRefineHead',
    num_classes=_num_classes,
    in_channels=256,
    stacked_convs=2,
    feat_channels=256,
    assign_by_circumhbbox=None,
    anchor_generator=dict(
        type='PseudoAnchorGenerator', strides=[8, 16, 32, 64, 128]),
    bbox_coder=dict(
        type='DeltaXYWHAOBBoxCoder',
        angle_range=_angle_version,
        norm_factor=1,
        edge_swap=False,
        proj_xy=True,
        target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
        target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
    loss_cls=dict(
        type='FocalLoss',
        use_sigmoid=True,
        gamma=2.0,
        alpha=0.25,
        loss_weight=1.0),
    loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
            iou_calculator=dict(type='RBboxOverlaps2D')),
        allowed_border=-1,
        pos_weight=-1,
        debug=False
    ),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=2000
    )
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
model = dict(
    builder = "few_shot_s2anet",
    type='S2ANet',
    backbone=_backbone,
    neck=_neck,
    fam_head=_fam_head,
    align_cfgs=dict(
        type='AlignConv',
        kernel_size=3,
        channels=256,
        featmap_strides=[8, 16, 32, 64, 128]),
    odm_head=_odm_head,
    modules = dict(

        s2anet = [
            "mmrotate/models/detectors/s2anet.py",
            "mmrotate/models/dense_heads/rotated_retina_head.py",
            "mmdet/models/losses/focal_loss.py",
            "mmdet/models/losses/smooth_l1_loss.py",
            "mmrotate/models/dense_heads/odm_refine_head.py",
            "mmrotate/core/anchor/anchor_generator.py",
        ]
    ),
)

weights = dict(
    backbone_pretrained = 'torchvision://resnet50',
)