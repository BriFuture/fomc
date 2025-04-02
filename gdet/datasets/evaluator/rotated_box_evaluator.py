import os, time, re
import os.path as osp

from .basic_evaluator import BasicEvaluator
from ..dataset.custom_dataset import CustomDataset
from gdet.registries import EVALUATORS
from mmrotate.core import eval_rbbox_map
from gdet.ops.nms_rotated import nms_rotated

@EVALUATORS.register_module()
class RotatedBoxEvaluator(BasicEvaluator):
    def __init__(self, config, dataset: "CustomDataset"):
        super().__init__()
        self.m_dataset = dataset
    

    def task_update(self, output):
        pass

    def evaluate(self,
                 results: "dict",
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None,
                 nproc=4):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
            nproc (int): Processes used for computing TP and FP.
                Default: 4.
        """
        nproc = min(nproc, os.cpu_count())
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.m_dataset.get_ann_info(i) for i in range(len(self.m_dataset))]
        filenames = [osp.basename(di['filename']) for di in self.m_dataset.m_data_infos]
        assert len(annotations) == len(results) and list(results.keys()) == filenames
        result_list = list(results.values())

        eval_results = {}
        dst_classes = self.m_dataset.dst_classes
        if metric == 'mAP':
            assert isinstance(iou_thr, float)
            mean_ap, _ = eval_rbbox_map(
                result_list,
                annotations,
                scale_ranges=scale_ranges,
                iou_thr=iou_thr,
                dataset=dst_classes,
                logger=logger,
                nproc=nproc)
            eval_results['mAP'] = mean_ap
        else:
            raise NotImplementedError

        return eval_results
    
