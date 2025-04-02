
from bfcommon.data_container import DataContainer
from gdet.structures.datasets import *
from gdet.registries import PIPELINES

@PIPELINES.register_module()
class TF_Collect(object):
    """TF_Collect data from the loader relevant to the specific task.

    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "proposals", "gt_bboxes",
    "gt_bboxes_ignore", "gt_labels", and/or "gt_masks".

    The "img_meta" item is always populated.  The contents of the "img_meta"
    dictionary depends on "meta_keys". By default this includes:

        - "img_shape": shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - "scale_factor": a float indicating the preprocessing scale

        - "flip": a boolean indicating if image flip transform was used

        - "filename": path to the image file

        - "ori_shape": original shape of the image as a tuple (h, w, c)

        - "pad_shape": image shape after padding

        - "img_norm_cfg": a dict of normalization information:

            - mean - per channel mean subtraction
            - std - per channel std divisor
            - to_rgb - bool indicating if bgr was converted to rgb

    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ``('filename', 'ori_filename', 'ori_shape', 'img_shape',
            'pad_shape', 'scale_factor', 'flip', 'flip_direction',
            'img_norm_cfg')``
    """

    def __init__(self, keys:"list[str]", meta_keys:"list[str]"=None, additional_meta_keys=None):
        self.keys = keys
        if meta_keys is None:
            meta_keys = ('filename', 'ori_filename', 'ori_shape', 'img_shape', 
                        'pad_shape', 'scale_factor', 'flip',
                        'flip_direction', 'img_norm_cfg', "ids")
        if additional_meta_keys is not None:
            meta_keys = tuple(list(meta_keys) + additional_meta_keys)
        self.meta_keys = set(meta_keys)
        
    def __call__(self, results: "DataItemsAfterBundle"):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:mmcv.DataContainer.

        Args:
            results (dict): Result dict contains the data to collect.

        Returns:
            dict: The result dict contains the following keys

                - keys in``self.keys``
                - ``img_metas``
        """
        data = DataTransedItems()
        img_meta = ImageMeta()
        inter_keys = self.meta_keys.intersection(results.keys())
        for key in inter_keys:
            img_meta[key] = results[key]

        data['img_metas'] = DataContainer(img_meta, cpu_only=True)
        for key in self.keys:
            if key in results:
                data[key] = results[key]
        return data
        

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys}, meta_keys={self.meta_keys})'
