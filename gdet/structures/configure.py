from typing import TypedDict

class DataLoaderConfigType(TypedDict):
    pin_memory: bool
    shuffle: bool
    sampling_strategy: str
    samples_per_gpu: int
    workers_per_gpu: int

class TrainConfigType(TypedDict):
    criterion: str
    loss_weight: float
    loss_weight_decay_rate: float
    data_loader: dict
    solver: dict

class DatasetConfigType(TypedDict):
    train: "dict"
    val: "dict"
    test: "dict"

class SolverConfigType(TypedDict):
    optimizer: str
    warmup_steps: int
    weight_decay: float
    momentum: float
    nesterov: float
    adamw_betas: float
    learning_rate: float

class ExpConfigType(TypedDict):
    type: "str"
    epochs: int
    load_from: str

class BaseConfigType(TypedDict):
    pass

class ModelRPN_LabelConfigType(TypedDict):
    rpn_match_threshold: float
    rpn_unmatched_threshold: float
    rpn_batch_size_per_im: int
    rpn_fg_fraction: float

class ModelCommon_ConfigType(TypedDict):
    start_level: int
    end_level: int
    
class ModelRPN_ConfigType(TypedDict):
    anchors_per_location: int
    num_convs: int

class ModelAnchorConfigType(TypedDict):
    num_scales: int
    aspect_ratios: "list[float]"
    anchor_size : "list[int]"

class ModelConfigType(TypedDict):
    rpn: ModelRPN_ConfigType
    anchor: ModelAnchorConfigType
    anchor_label: ModelRPN_LabelConfigType
    vision_model: "dict"

class DistConfigType(TypedDict):
    gpu_ids: list
    gpu_nums: int
    distributed: bool
    launcher: str
    port: int

class ConfigType(TypedDict):
    train: TrainConfigType
    val: TrainConfigType
    test: TrainConfigType
    solver: SolverConfigType
    weights: "dict"
    experiment: ExpConfigType

    model: ModelConfigType

    ALL_CLASSES: list
    dataset: DatasetConfigType
    dist: DistConfigType


class ExpTrainConfigType(TypedDict):

    def init(self): ...
    def train(self): ...
    def evaluate_val(self): ...
    def merge_args(self, args): ...