from .loaders import LoadImageFromFile, LoadAnnotations
from .formatting import ImageToTensor, DefaultFormatBundle
from .transforms import TF_Normalize, TF_Resize, TF_ResizeMaskSeg, TF_Pad
from .collection import TF_Collect
from .test_time_aug import TF_MultiScaleFlipAug
from torchvision.transforms import ToTensor

from gdet.registries import PIPELINES
from torchvision.transforms import Compose
PIPELINES.register_module(name="Compose", module=Compose)
PIPELINES.register_module(name="ToTensor", module=ToTensor)

from .builder import construct_transforms
