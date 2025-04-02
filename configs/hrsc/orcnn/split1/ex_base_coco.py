_base_ = "ex_base.py"

dataset = dict(
    evaluator = dict(
        type="RotatedCocoEvaluator",
        ann_file = "datasets/HRSC2016/FullDataSet/coco_annos/val.json"
    ),
    modules = dict(
        coco = [
            "gdet/datasets/evaluator/coco.py",
            "gdet/datasets/evaluator/rotated_coco.py"
        ]
    )
)

experiment = dict(
    use_cache_detection_result = False,
)