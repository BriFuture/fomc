_base_ = ["../model_base.py"]

model = dict(
    modules = dict(
        few_shot=["gdet.modules.few_shot"],
        mmdet = [            
            "mmdet/models/backbones/resnet.py",
            "mmdet/models/necks/fpn.py",
            "mmdet/core/bbox/assigners/max_iou_assigner.py",
            "mmdet/core/bbox/samplers/pseudo_sampler.py",
        ],
    )
)
