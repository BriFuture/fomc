from gdet.registries import PIPELINES
from torchvision.transforms import Compose
def construct_transforms(cfg):
    cfg = cfg.copy()
    transform_list = []
    for tr_cfg in cfg:
        cls_type = tr_cfg.pop("type")
        t_cls = PIPELINES.get(cls_type)
        assert t_cls is not None, cls_type
        transform_list.append(t_cls(**tr_cfg))

    return Compose(transform_list)