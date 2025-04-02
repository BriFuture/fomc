import torch
import torch.nn as nn
import torch.nn.functional as F

from gdet.registries import MODEL_BACKBONES as NECKS
from mmdet.models.necks.fpn import FPN
import logging
logger = logging.getLogger("gdet.models.fsfpn")
@NECKS.register_module()
class FsFPN(FPN):
    """与 FPN 类似，但是通过 frozen 参数可以设置是否冻结其中一些 layer
    """
    def __init__(self, *args, **kwargs):
        frozen = kwargs.pop("frozen", None)
        defroze_steps = kwargs.pop("defroze_steps", -1)
        super(FsFPN, self).__init__(*args, **kwargs)

        self.defroze_steps = defroze_steps
        self.freeze_by_layer(frozen)

    def freeze_by_layer(self, frozen: "list[int]"):
        self.frozen_layers = frozen
        if frozen is not None:
            self.set_freeze(frozen, True)

    def set_freeze(self, frozen_layers: "list[int]", freeze: "bool"):
        logger.info(f"Freeze fpn : {freeze} {frozen_layers}")
        grad = not freeze
        for i in frozen_layers:
            if i >= len(self.lateral_convs): 
                continue
            for param in self.lateral_convs[i].parameters():
                param.requires_grad = grad
        for i in frozen_layers:
            if i >= len(self.fpn_convs): 
                continue
            for param in self.fpn_convs[i].parameters():
                param.requires_grad = grad

    
