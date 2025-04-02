_base_ = ["model_fs_base.py"]
### from mmrotate/configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90.py
_angle_version = 'le90'
_num_classes = 15
_rpn_head = dict(
    type='OrientedRPNHead',
    in_channels=256,
    feat_channels=256,
    version=_angle_version,
    anchor_generator=dict(
        type='AnchorGenerator',
        scales=[8],
        ratios=[0.5, 1.0, 2.0],
        strides=[4, 8, 16, 32, 64]),
    bbox_coder=dict(
        type='MidpointOffsetCoder',
        angle_range=_angle_version,
        target_means=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        target_stds=[1.0, 1.0, 1.0, 1.0, 0.5, 0.5]),
    loss_cls=dict(
        type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
    loss_bbox=dict(
        type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)
)
_bbox_head = dict(
    type='RotatedShared2FCBBoxHead',
    in_channels=256,
    fc_out_channels=1024,
    roi_feat_size=7,
    num_classes=_num_classes,
    bbox_coder=dict(
        type='DeltaXYWHAOBBoxCoder',
        angle_range=_angle_version,
        norm_factor=None,
        edge_swap=True,
        proj_xy=True,
        target_means=(.0, .0, .0, .0, .0),
        target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)),
    reg_class_agnostic=True,
    loss_cls=dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)
)
_roi_head = dict(
    type='OrientedStandardRoIHead',
    bbox_roi_extractor=dict(
        type='RotatedSingleRoIExtractor',
        roi_layer=dict(
            type='RoIAlignRotated',
            output_size=7,
            sampling_ratio=2,
            clockwise=True),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=_bbox_head,
    version = _angle_version,
)
_train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            match_low_quality=True,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_pre=2000,
        max_per_img=2000,
        nms=dict(type='nms', iou_threshold=0.8),
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            match_low_quality=False,
            iou_calculator=dict(type='RBboxOverlaps2D'),
            ignore_iof_thr=-1),
        sampler=dict(
            type='RRandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False
    )
)
_test_cfg = dict(
    rpn=dict(
        nms_pre=2000,
        max_per_img=2000,
        nms=dict(type='nms', iou_threshold=0.8),
        min_bbox_size=0
    ),
    rcnn=dict(
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
    norm_cfg=dict(type='BN', requires_grad=True),
    norm_eval=True,
    style='pytorch',
    init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
)
_neck = dict(
    type='FPN',
    in_channels=[256, 512, 1024, 2048],
    out_channels=256,
    num_outs=5
)
model = dict(
    builder = "few_shot_orcnn",
    type='OrientedRCNN',
    backbone=_backbone,
    neck=_neck,
    rpn_head=_rpn_head,
    roi_head=_roi_head,
    train_cfg=_train_cfg,
    test_cfg=_test_cfg,
    modules = dict(
        orcnn = [
            "mmdet/models/losses/cross_entropy_loss.py",
            "mmdet/models/losses/smooth_l1_loss.py",
            "mmrotate/models/dense_heads/oriented_rpn_head.py",
            "mmrotate/core/anchor/anchor_generator.py",
            "mmrotate/models/roi_heads/roi_extractors/rotate_single_level_roi_extractor.py",
            "gdet/ops/roi_align_rotated.py",
            "mmrotate/models/roi_heads/bbox_heads/convfc_rbbox_head.py",
            "mmrotate/models/roi_heads/oriented_standard_roi_head.py",
            "mmrotate/models/detectors/oriented_rcnn.py",
            "mmdet/core/bbox/iou_calculators/iou2d_calculator.py",
            "mmdet/core/bbox/samplers/random_sampler.py",
        ]
    ),
)

weights = dict(
    backbone_pretrained = 'torchvision://resnet50',
)