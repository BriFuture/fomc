import os, os.path as osp
import numpy as np
import logging
from multiprocessing import get_context
from terminaltables import AsciiTable

from .dota import EVALUATORS, RotatedBoxEvaluator
from mmrotate.core.evaluation.eval_map import get_cls_results, tpfp_default, average_precision

g_logger = logging.getLogger("gdet.mmrotate.core.eval")
def print_map_summary(mean_ap, mean_bap, mean_nap,
                      results,
                      dataset,
                      scale_ranges=None,
                      ):
    """Print mAP and results of each class.

    A table will be printed to show the gts/dets/recall/AP of each class and
    the mAP.

    Args:
        mean_ap (float): Calculated from `eval_map()`.
        results (list[dict]): Calculated from `eval_map()`.
        dataset (list[str] | str | None): Dataset name or dataset classes.
        scale_ranges (list[tuple] | None): Range of scales to be evaluated.
    """


    if isinstance(results[0]['ap'], np.ndarray):
        num_scales = len(results[0]['ap'])
    else:
        num_scales = 1

    if scale_ranges is not None:
        assert len(scale_ranges) == num_scales

    num_classes = len(results)

    recalls = np.zeros((num_scales, num_classes), dtype=np.float32)
    aps = np.zeros((num_scales, num_classes), dtype=np.float32)
    num_gts = np.zeros((num_scales, num_classes), dtype=int)
    for i, cls_result in enumerate(results):
        if cls_result['recall'].size > 0:
            recalls[:, i] = np.array(cls_result['recall'], ndmin=2)[:, -1]
        aps[:, i] = cls_result['ap']
        num_gts[:, i] = cls_result['num_gts']

    assert dataset is not None
    # if dataset is None:
        # label_names = [str(i) for i in range(num_classes)]
    # else:
        # label_names = dataset
    label_names = dataset

    if not isinstance(mean_ap, list):
        mean_ap = [mean_ap]
    if not isinstance(mean_bap, list):
        mean_bap = [mean_bap]        
    if not isinstance(mean_nap, list):
        mean_nap = [mean_nap]
    header = ['idx', 'class', 'gts', 'dets', 'recall', 'ap']
    for i in range(num_scales):
        if scale_ranges is not None:
            g_logger.info(f'Scale range {scale_ranges[i]}')
        table_data = [header]
        for ci, j in enumerate(range(num_classes)):
            row_data = [
                ci,
                label_names[j], num_gts[i, j], results[j]['num_dets'],
                f'{recalls[i, j]:.3f}', f'{aps[i, j]:.3f}'
            ]
            table_data.append(row_data)
        table_data.append(['mAP', '', '', '', '', f'{mean_ap[i]:.3f}'])
        table_data.append(['bAP', '', '', '', '', f'{mean_bap[i]:.3f}'])
        table_data.append(['nAP', '', '', '', '', f'{mean_nap[i]:.3f}'])
        table = AsciiTable(table_data)
        target_row_index = - 3

        # 创建一个分隔符行（与表格宽度一致）
        separator = ['─' * len(cell) for cell in table_data[0]]  # 根据每列的宽度生成分隔符
        # 在倒数第三行的下方插入分隔符
        table_data.insert(target_row_index, separator)
        # table.inner_footing_row_border = True
        g_logger.info('\n' + table.table)

def eval_rbbox_map(det_results,
                   annotations,
                   scale_ranges=None,
                   iou_thr=0.5,
                   use_07_metric=True,
                   dataset=None,
                   base_classes=None,
                   logger=None,
                   nproc=4):
    """Evaluate mAP of a rotated dataset.

    Args:
        det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotations (list[dict]): Ground truth annotations where each item of
            the list indicates an image. Keys of annotations are:

            - `bboxes`: numpy array of shape (n, 5)
            - `labels`: numpy array of shape (n, )
            - `bboxes_ignore` (optional): numpy array of shape (k, 5)
            - `labels_ignore` (optional): numpy array of shape (k, )
        scale_ranges (list[tuple] | None): Range of scales to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. A range of
            (32, 64) means the area range between (32**2, 64**2).
            Default: None.
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        use_07_metric (bool): Whether to use the voc07 metric.
        dataset (list[str] | str | None): Dataset name or dataset classes,
            there are minor differences in metrics for different datasets, e.g.
            "voc07", "imagenet_det", etc. Default: None.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmcv.utils.print_log()` for details. Default: None.
        nproc (int): Processes used for computing TP and FP.
            Default: 4.

    Returns:
        tuple: (mAP, [dict, dict, ...])
    """
    assert len(det_results) == len(annotations)

    num_imgs = len(det_results)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    ### add num_class
    num_classes = len(det_results[0])  # positive class num
    area_ranges = ([(rg[0]**2, rg[1]**2) for rg in scale_ranges]
                   if scale_ranges is not None else None)

    pool = get_context('spawn').Pool(nproc)
    eval_results = []
    for i in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts, cls_gts_ignore = get_cls_results(
            det_results, annotations, i)

        # compute tp and fp for each image with multiple processes
        tpfp = pool.starmap(
            tpfp_default,
            zip(cls_dets, cls_gts, cls_gts_ignore,
                [iou_thr for _ in range(num_imgs)],
                [area_ranges for _ in range(num_imgs)]))
        tp, fp = tuple(zip(*tpfp))
        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts = np.zeros(num_scales, dtype=int)
        for _, bbox in enumerate(cls_gts):
            if area_ranges is None:
                num_gts[0] += bbox.shape[0]
            else:
                gt_areas = bbox[:, 2] * bbox[:, 3]
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum((gt_areas >= min_area)
                                         & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        num_dets = cls_dets.shape[0]
        sort_inds = np.argsort(-cls_dets[:, -1])
        tp = np.hstack(tp)[:, sort_inds]
        fp = np.hstack(fp)[:, sort_inds]
        # calculate recall and precision with tp and fp
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
        # calculate AP
        if scale_ranges is None:
            recalls = recalls[0, :]
            precisions = precisions[0, :]
            num_gts = num_gts.item()
        mode = 'area' if not use_07_metric else '11points'
        ap = average_precision(recalls, precisions, mode)
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'ap': ap
        })
    pool.close()
    assert dataset is not None
    if scale_ranges is not None:
        # shape (num_classes, num_scales)
        all_ap = np.vstack([cls_result['ap'] for cls_result in eval_results])
        all_num_gts = np.vstack(
            [cls_result['num_gts'] for cls_result in eval_results])
        mean_ap = []
        mean_bap = []
        mean_nap = []
        for i in range(num_scales):
            if np.any(all_num_gts[:, i] > 0):
                mean_ap.append(all_ap[all_num_gts[:, i] > 0, i].mean())
            else:
                mean_ap.append(0.0)
    else:
        aps = []
        baps = []
        naps = []
        for idx, cls_result in enumerate(eval_results):
            if cls_result['num_gts'] > 0:
                aps.append(cls_result['ap'])
                if dataset[idx] in base_classes:
                    baps.append(cls_result['ap'])
                else:
                    naps.append(cls_result['ap'])

        mean_ap = np.array(aps).mean().item() if aps else 0.0
        mean_bap = np.array(baps).mean().item() if baps else 0.0
        mean_nap = np.array(naps).mean().item() if naps else 0.0

    if logger != 'silent':
        print_map_summary(mean_ap, mean_bap, mean_nap, eval_results, dataset, area_ranges,)

    return mean_ap, mean_bap, mean_nap, eval_results

@EVALUATORS.register_module()
class FS_RotatedBoxEvaluator(RotatedBoxEvaluator):
    def __init__(self, config, dataset: "DOTADataset"):
        self.m_dataset = dataset
        pass
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
        assert len(annotations) == len(results) 
        ### 当采用 dist eval 时候，需要将 results.keys 和 filenames 对应
        assert list(results.keys()) == filenames
        result_list = list(results.values())

        eval_results = {}
        base_classes = list(self.m_dataset.base_classes)
        all_classes = list(self.m_dataset.all_classes)
        if metric == 'mAP':
            assert isinstance(iou_thr, float)
            mean_ap, mean_bap, mean_nap, _ = eval_rbbox_map(
                result_list,
                annotations,
                scale_ranges=scale_ranges,
                iou_thr=iou_thr,
                dataset=all_classes,
                base_classes = base_classes,
                logger=logger,
                nproc=nproc)
            eval_results['mAP'] = mean_ap
            eval_results['bAP'] = mean_bap
            eval_results['nAP'] = mean_nap
        else:
            raise NotImplementedError

        return eval_results
    