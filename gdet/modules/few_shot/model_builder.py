from gdet.registries import MODULE_BUILD_FUNCS, MODELS, MODEL_BACKBONES, MODEL_HEADS, LOSSES
from gdet.structures.configure import ConfigType
from mmrotate.models.detectors.utils import AlignConvModule
from mmcv.utils.logging import logger_initialized
from gdet.utils.builder_utils import build_with_registry

@MODULE_BUILD_FUNCS.register_module(name="few_shot_s2anet")
def build_s2anet_model(config: ConfigType):
    model_cfg = config.model.clone()
    detector_type = model_cfg.pop("type")


    backbone_cfg = model_cfg['backbone']
    backbone = build_with_registry(MODEL_BACKBONES, backbone_cfg)

    neck_cfg = model_cfg['neck']
    neck = build_with_registry(MODEL_BACKBONES, neck_cfg)

    fam_head_cfg = model_cfg['fam_head']
    fam_head = build_with_registry(MODEL_HEADS, fam_head_cfg)
    fam_head.build_loss(fam_head_cfg["loss_cls"], fam_head_cfg["loss_bbox"], )

    align_cfgs = model_cfg['align_cfgs']
    align_conv_type = align_cfgs['type']
    align_conv_size = align_cfgs['kernel_size']
    feat_channels = align_cfgs['channels']
    featmap_strides = align_cfgs['featmap_strides']
    if align_conv_type == 'AlignConv':
        align_conv = AlignConvModule(feat_channels,
                                            featmap_strides,
                                            align_conv_size)
    else:
        align_conv = None
    # pass

    # fam_head.update(test_cfg=test_cfg)
    # odm_head.update(test_cfg=test_cfg)
    odm_head_cfg = model_cfg['odm_head']
    odm_head = build_with_registry(MODEL_HEADS, odm_head_cfg)
    odm_head.build_loss(odm_head_cfg["loss_cls"], odm_head_cfg["loss_bbox"], )


    detector_cls = MODELS.get(detector_type)
    assert detector_cls is not None, detector_type
    # few_shot_train=model_cfg.get("few_shot_train", False)
    detector = detector_cls(backbone, neck=neck, fam_head=fam_head, align_conv=align_conv, odm_head=odm_head, config=model_cfg)
    if hasattr(detector, "init_weights"):
        logger_initialized['silent'] = None
        detector.init_weights()
        logger_initialized.pop('silent', None)
    return detector

@MODULE_BUILD_FUNCS.register_module(name="few_shot_orcnn")
def build_orcnn_model(config: ConfigType):
    model_cfg = config.model.clone()
    detector_type = model_cfg.pop("type")


    backbone_cfg = model_cfg['backbone']
    backbone = build_with_registry(MODEL_BACKBONES, backbone_cfg)

    neck_cfg = model_cfg['neck']
    neck = build_with_registry(MODEL_BACKBONES, neck_cfg)

    train_cfg = model_cfg.train_cfg
    test_cfg = model_cfg.test_cfg

    rpn_head_cfg = model_cfg['rpn_head']
    rpn_head_cfg.update(train_cfg=train_cfg.rpn, test_cfg=test_cfg.rpn)
    rpn_head = build_with_registry(MODEL_HEADS, rpn_head_cfg)
    rpn_head.build_loss(rpn_head_cfg["loss_cls"], rpn_head_cfg["loss_bbox"], )
    
    roi_head_cfg = model_cfg['roi_head']

    
    bbox_roi_extractor_cfg = roi_head_cfg.pop("bbox_roi_extractor")
    bbox_roi_extractor = build_with_registry(MODEL_HEADS, bbox_roi_extractor_cfg)

    bbox_head_cfg = roi_head_cfg.pop("bbox_head")
    bbox_head = build_with_registry(MODEL_HEADS, bbox_head_cfg)
    bbox_head.build_loss(bbox_head_cfg["loss_cls"], bbox_head_cfg["loss_bbox"],)
    bbox_head.init()

    roi_head_cfg.update(train_cfg=train_cfg.rcnn, test_cfg=test_cfg.rcnn)

    roi_head_cls_type = roi_head_cfg.pop("type")
    roi_head_cls = MODEL_HEADS.get(roi_head_cls_type)
    assert roi_head_cls is not None, roi_head_cls_type
    roi_head = roi_head_cls(bbox_roi_extractor, bbox_head, **roi_head_cfg)
    # roi_head.build_loss(roi_head_cfg["loss_cls"], roi_head_cfg["loss_bbox"], )


    detector_cls = MODELS.get(detector_type)
    assert detector_cls is not None, detector_type
    detector = detector_cls(backbone, neck=neck, rpn_head=rpn_head, roi_head=roi_head,
                            train_cfg=model_cfg.train_cfg, test_cfg=model_cfg.test_cfg, config=model_cfg)
    if hasattr(detector, "init_weights"):
        logger_initialized['silent'] = None
        detector.init_weights()
        logger_initialized.pop('silent', None)

    return detector

@MODULE_BUILD_FUNCS.register_module(name="r3det")
def build_s2anet_model(config: ConfigType):
    model_cfg = config.model.clone()
    detector_type = model_cfg.pop("type")


    backbone_cfg = model_cfg['backbone']
    backbone = build_with_registry(MODEL_BACKBONES, backbone_cfg)

    neck_cfg = model_cfg['neck']
    neck = build_with_registry(MODEL_BACKBONES, neck_cfg)


    bbox_head_cfg = model_cfg['bbox_head']
    bbox_head = build_with_registry(MODEL_HEADS, bbox_head_cfg)
    bbox_head.build_loss(bbox_head_cfg["loss_cls"], bbox_head_cfg["loss_bbox"], )

    refine_heads_cfg_list: "dict" = model_cfg['refine_heads']
    refine_heads = []
    for refine_heads_cfg in refine_heads_cfg_list.values():
        refine_head = build_with_registry(MODEL_HEADS, refine_heads_cfg)
        refine_head.build_loss(refine_heads_cfg["loss_cls"], refine_heads_cfg["loss_bbox"], )
        
        refine_heads.append(refine_head)

    detector_cls = MODELS.get(detector_type)
    assert detector_cls is not None, detector_type
    # few_shot_train=model_cfg.get("few_shot_train", False)
    detector = detector_cls(model_cfg.num_refine_stages, 
                            backbone=backbone, neck=neck, bbox_head=bbox_head, refine_heads=refine_heads, config=model_cfg, 
                            train_cfg=model_cfg.train_cfg, test_cfg=model_cfg.test_cfg, frm_cfgs=model_cfg.frm_cfgs)
    if hasattr(detector, "init_weights"):
        logger_initialized['silent'] = None
        detector.init_weights()
        logger_initialized.pop('silent', None)
    return detector