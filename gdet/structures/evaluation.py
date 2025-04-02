from typing import TypedDict
import numpy as np
class CocoPredAnn(TypedDict):
    image_id: int
    category_id: int
    bbox: np.ndarray
    score: float
    id: int

import numpy as np
from typing import TypedDict

class ModelPredictions(TypedDict):
    detection_boxes: "list[np.ndarray]"
    detection_classes: "list[np.ndarray]"
    detection_scores: "list[np.ndarray]"
    image_info: "list[np.ndarray]"
    source_id: "list[np.ndarray]"
    num_detections: "list[np.ndarray]"
    ## not needed when evaluation
    box_outputs: "list[np.ndarray]"
    class_outputs: "list[np.ndarray]"
    rpn_box_outputs: "list[DataBatchBoxTargets]"
    rpn_score_outputs: "list[DataBatchBoxTargets]"


class ModelCocoOutput(TypedDict):
    detection: "ModelPredictions"


class Dino_Coco_output(TypedDict):
    pass