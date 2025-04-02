_base_ = ["model_base.py"]

model = dict(
    rpn = dict(
        num_convs=2,
        train_pre_nms_top_k = 2000,
        train_post_nms_top_k = 1000,
        val_pre_nms_top_k = 500,
        val_post_nms_top_k = 250,
        nms_threshold = 0.7,
    ),
    roi_head = dict(
        num_heads = 32,
        output_dim = 1024,
        spacial_dim=7,
        embed_dim=2048,
    ),
    fastrcnn = dict(
        type="FastrcnnHead",
        num_classes = 1024,
        fc_dims = 1024,
        num_fcs = 2,
        num_convs = 4,
        num_filters = 256,
        class_box_regression = False,
        use_class_bias = False,
        use_batch_norm = True,
        mix_gt_boxes = True,
        allow_low_quality = False,
        fg_iou_threshold = 0.5,
    ),
    faster_rcnn = dict(
        batch_norm_group_size = 0,
        temperature_scale = 0.1,
        base_vlm_weight = 0.35,
        novel_vlm_weight = 0.65,
        use_frozen_vlm = False,
        objectness_weight = 0.0,
        output_dec_boxes = False,
        clip_sim_temp = 0.01,
        roi_scale_factor = None,
        include_top_level_during_eval = True,
    )
)