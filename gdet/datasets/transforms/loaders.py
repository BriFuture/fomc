import os, os.path as osp
from PIL import Image
import numpy as np
import pycocotools.mask as maskUtils
from gdet.structures.datasets import *
from gdet.registries import PIPELINES
from gdet.datasets.image.mask import BitmapMasks, PolygonMasks

@PIPELINES.register_module()
class LoadImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self, to_float32=True, color_type='color', extra_property=None):
        self.to_float32 = to_float32
        self.color_type = color_type
        if extra_property is None:
            self.extra_property = {}
        else:
            self.extra_property = extra_property

    def __call__(self, results: "DataBatchItems"):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        im_info = results['img_info']
        ori_filename = im_info['filename']
        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'], ori_filename)
        else:
            filename = ori_filename

        img = Image.open(filename)
        img = img.convert('RGB')
        img = np.asarray(img)
        # if self.to_float32:
        img = img.astype(np.float32)
        img /= 255
        # if img is None:
        #     print(filename)
        results['filename'] = filename
        # results['ori_filename'] = ori_filename
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        results: "DataItemsAfterLoad"
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations(object):
    """Load mutiple types of annotations.

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
             Default: True.
        with_label (bool): Whether to parse and load the label annotation.
            Default: True.
        with_mask (bool): Whether to parse and load the mask annotation.
             Default: False.
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Default: False.
        poly2mask (bool): Whether to convert the instance masks from polygons
            to bitmaps. Default: True.
    """

    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=False,
                 with_seg=False,
                 with_id=False,
                 poly2mask=True,
                 with_word=False,
                 **kwargs):
        self.with_bbox = with_bbox
        self.with_ignore_bbox = kwargs.get("with_ignore_bbox", False)
        self.with_label = with_label
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.with_id = with_id
        self.with_word = with_word
        self.poly2mask = poly2mask
        self.file_client = None
        self._max_detections = 0 ## 若小于等于0,则对 box 进行填充，按 0 填充

    def _load_bboxes(self, results):
        """Private function to load bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box annotations.
        """

        ann_info: "AnnInfo" = results['ann_info']
        boxes = ann_info['bboxes']
        if self._max_detections > 0:
            zero = np.zeros((self._max_detections, 4), dtype=boxes.dtype)
            num_box = boxes.shape[0]
            zero[:num_box, :] = boxes
            results['gt_bboxes'] = zero
        else:
            gt_boxes = boxes.copy()
            # ### bsf.c remove tiny objects
            # w = gt_boxes[:, 2] - gt_boxes[:, 0]
            # h = gt_boxes[:, 3] - gt_boxes[:, 1]
            # mask = (w >= 8) & (h >= 8)
            # results['gt_bboxes'] = gt_boxes[mask]
            results['gt_bboxes'] = gt_boxes
            
        results['bbox_fields'].append('gt_bboxes')

        gt_bboxes_ignore = ann_info.get('bboxes_ignore', None)
        if gt_bboxes_ignore is not None and self.with_ignore_bbox:
            results['gt_bboxes_ignore'] = gt_bboxes_ignore.copy()
            results['bbox_fields'].append('gt_bboxes_ignore')
        return results
    
    def _load_words(self, results):
        """Private function to load word.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded word.
        """

        ann_info = results['ann_info']
        results['words'] = ann_info['word'].clone()
        return results

    def _load_ids(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """
        ids = results['ann_info']['ids']
        results['ids'] = ids.copy()
        return results
    
    def _load_labels(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """
        labels: "np.ndarray" = results['ann_info']['labels']
        if self._max_detections > 0:
            zero = np.zeros((self._max_detections,), dtype=labels.dtype) ## 0 as background
            num_box = labels.shape[0]
            zero[:num_box] = labels
            results['gt_labels'] = zero
        else:
            results['gt_labels'] = labels.copy()
        return results


    def process_polygons(self, polygons):
        """Convert polygons to list of ndarray and filter invalid polygons.

        Args:
            polygons (list[list]): Polygons of one instance.

        Returns:
            list[numpy.ndarray]: Processed polygons.
        """

        polygons = [np.array(p) for p in polygons]
        valid_polygons = []
        for polygon in polygons:
            if len(polygon) % 2 == 0 and len(polygon) >= 6:
                valid_polygons.append(polygon)
        return valid_polygons


    def __call__(self, results: "DataItemsAfterLoad"):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """

        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_word:
            results = self._load_words(results)
        if self.with_id:
            results = self._load_ids(results)
        if self.with_label:
            results = self._load_labels(results)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_id={self.with_id}, '
        return repr_str

@PIPELINES.register_module()
class SegAnnotationLoader(LoadAnnotations):

    def _poly2mask(self, mask_ann, img_h, img_w):
        """Private function to convert masks represented with polygon to
        bitmaps.

        Args:
            mask_ann (list | dict): Polygon mask annotation input.
            img_h (int): The height of output mask.
            img_w (int): The width of output mask.

        Returns:
            numpy.ndarray: The decode bitmap mask of shape (img_h, img_w).
        """

        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask
    def _load_masks(self, results):
        """Private function to load mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded mask annotations.
                If ``self.poly2mask`` is set ``True``, `gt_mask` will contain
                :obj:`PolygonMasks`. Otherwise, :obj:`BitmapMasks` is used.
        """

        h, w = results['img_info']['height'], results['img_info']['width']
        gt_masks = results['ann_info']['masks']
        if self.poly2mask:
            gt_masks = BitmapMasks(
                [self._poly2mask(mask, h, w) for mask in gt_masks], h, w)
        else:
            gt_masks = PolygonMasks(
                [self.process_polygons(polygons) for polygons in gt_masks], h,
                w)
        results['gt_masks'] = gt_masks
        results['mask_fields'].append('gt_masks')
        return results

    def _load_semantic_seg(self, results):
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """


        filename = osp.join(results['seg_prefix'],
                            results['ann_info']['seg_map'])
        im = Image.open(filename)
        im = np.array(im)
        results['gt_semantic_seg'] = im.squeeze()
        results['seg_fields'].append('gt_semantic_seg')
        return results
    
    def __call__(self, results: "DataItemsAfterLoad"):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """
        results = super()(results)

        if self.with_mask:
            results = self._load_masks(results)
        if self.with_seg:
            results = self._load_semantic_seg(results)
        return results    

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_id={self.with_id}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_seg={self.with_seg})'
        repr_str += f'poly2mask={self.poly2mask})'
        return repr_str
    
