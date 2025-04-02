import numpy as np
class BasicEvaluator():
    def __init__(self, need_rescale_bboxes=True):
        self._required_prediction_fields = [
            "source_id",
            "num_detections",
            "detection_classes",
            "detection_scores",
            "detection_boxes",
        ]
        self._need_rescale_bboxes = need_rescale_bboxes
        if self._need_rescale_bboxes:
            self._required_prediction_fields.append("image_info")
        self._predictions = {}
    
    def reset(self):
        """Resets internal states for a fresh run."""
        self._predictions = {}


    @staticmethod
    def _process_predictions(predictions: dict):
        is_info = predictions["image_info"][:, 2:3, :]
        image_scale = np.tile(is_info, (1, 1, 2))
        predictions["detection_boxes"] = predictions["detection_boxes"].astype(
            np.float32
        )
        predictions["detection_boxes"] /= image_scale
        if "detection_outer_boxes" in predictions:
            predictions["detection_outer_boxes"] = predictions[
                "detection_outer_boxes"
            ].astype(np.float32)
            predictions["detection_outer_boxes"] /= image_scale

    def remove_unnecessary_keys(self, predictions: "dict"):
        if "detection" in predictions:
            predictions = predictions["detection"]  # Get task scope.
        ### remove unnecessary keys
        unnecessary_keys = []
        
        for k in predictions.keys():
            if k not in self._required_prediction_fields:
                unnecessary_keys.append(k)
        for k in unnecessary_keys:
            predictions.pop(k)
                