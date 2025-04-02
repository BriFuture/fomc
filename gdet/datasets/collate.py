from typing import Sequence, Mapping
import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate as torch_default_collate
from bfcommon.data_container import DataContainer

def collate(batch, samples_per_gpu=1):
    """Puts each data field into a tensor/DataContainer with outer dimension
    batch size.

    Extend default_collate to add support for
    :type:`~mmcv.parallel.DataContainer`. There are 3 cases.

    1. cpu_only = True, e.g., meta data
    2. cpu_only = False, stack = True, e.g., images tensors
    3. cpu_only = False, stack = False, e.g., gt bboxes
    """

    if not isinstance(batch, Sequence):
        raise TypeError(f'{batch.dtype} is not supported.')

    if isinstance(batch[0], DataContainer):
        return collate_data_container(batch, samples_per_gpu)
    elif isinstance(batch[0], Sequence):
        transposed = zip(*batch)
        return [collate(samples, samples_per_gpu) for samples in transposed]
    elif isinstance(batch[0], Mapping):
        retd = {}
        for key in batch[0].keys():
            dat = [d[key] for d in batch]
            retd[key] = collate(dat, samples_per_gpu)

        return retd
    else:
        return torch_default_collate(batch)

def collate_data_container(batch: "list[DataContainer]", samples_per_gpu: "int"):
    """将 list of DataContainer 进行 collate 操作
    """
    b0 = batch[0]
    if b0.cpu_only:
        ### img meta
        stacked = []
        for i in range(0, len(batch), samples_per_gpu):
            stacked.append([sample.data for sample in batch[i:i + samples_per_gpu]])
        return DataContainer(stacked, b0.stack, b0.padding_value, cpu_only=True)
    elif b0.stack:
        ## image tensors
        stacked = []
        for i in range(0, len(batch), samples_per_gpu):
            batch_i: "DataContainer" = batch[i]
            assert isinstance(batch_i.data, torch.Tensor)

            if batch_i.pad_dims is not None:
                ndim = batch_i.dim()
                assert ndim > batch_i.pad_dims
                max_shape = [0 for _ in range(batch_i.pad_dims)]
                for dim in range(1, batch_i.pad_dims + 1):
                    max_shape[dim - 1] = batch_i.size(-dim)
                for sample in batch[i:i + samples_per_gpu]:
                    for dim in range(0, ndim - batch_i.pad_dims):
                        assert batch_i.size(dim) == sample.size(dim)
                    for dim in range(1, batch_i.pad_dims + 1):
                        max_shape[dim - 1] = max(max_shape[dim - 1], sample.size(-dim))
                padded_samples = []
                for sample in batch[i:i + samples_per_gpu]:
                    pad = [0 for _ in range(batch_i.pad_dims * 2)]
                    for dim in range(1, batch_i.pad_dims + 1):
                        pad[2 * dim - 1] = max_shape[dim - 1] - sample.size(-dim)
                    padded_sample = F.pad(sample.data, pad, value=sample.padding_value)
                    padded_samples.append(padded_sample)
                ps: "torch.Tensor" = torch_default_collate(padded_samples)
                stacked.append(ps)
            elif batch_i.pad_dims is None:
                sample_data = [sample.data for sample in batch[i:i + samples_per_gpu]]
                ps: "torch.Tensor" = torch_default_collate(sample_data)
                stacked.append(ps)
            else:
                raise ValueError('pad_dims should be either None or integers (1-3)')
        return DataContainer(stacked, b0.stack, b0.padding_value)    
    else:
        # gt bboxes gt labels
        stacked = []
        for i in range(0, len(batch), samples_per_gpu):
            stacked.append([sample.data for sample in batch[i:i + samples_per_gpu]])
        return DataContainer(stacked, b0.stack, b0.padding_value)    