from bfcommon.registry import Registry, ExRegistry

DATA_SAMPLERS = Registry("data samplers")
FUNCTIONS = Registry('function')
DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline (DATASET_TRANSFORMS)')
EVALUATORS = Registry("dataset evaluators")

VISER = Registry('viser')

### usually framework
MODELS = Registry('models')
LOSSES = Registry('losses')
MODEL_HEADS = Registry('model_heads')
MODEL_BACKBONES = Registry('model_backbones')

MODEL_OPS = Registry('torch ops for some layer')

ROI_EXTRACTORS = Registry('roi extractors')
TASK_UTILS = Registry('task_utils')

MODULE_BUILD_FUNCS = ExRegistry("model build functions")

### class meta
DATA_METAS = ExRegistry('metas')
EXPS = Registry("exp")