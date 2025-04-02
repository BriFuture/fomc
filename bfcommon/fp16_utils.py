import functools
import warnings
from collections import abc
from inspect import getfullargspec

import numpy as np
import torch
import torch.nn as nn

# from .dist_utils import allreduce_grads as _allreduce_grads


def cast_tensor_type(inputs, src_type, dst_type):
    """Recursively convert Tensor in inputs from src_type to dst_type.

    Args:
        inputs: Inputs that to be casted.
        src_type (torch.dtype): Source type..
        dst_type (torch.dtype): Destination type.

    Returns:
        The same type with inputs, but all contained Tensors have been cast.
    """
    if isinstance(inputs, torch.Tensor):
        return inputs.to(dst_type)
    elif isinstance(inputs, str):
        return inputs
    elif isinstance(inputs, np.ndarray):
        return inputs
    elif isinstance(inputs, abc.Mapping):
        return type(inputs)({
            k: cast_tensor_type(v, src_type, dst_type)
            for k, v in inputs.items()
        })
    elif isinstance(inputs, abc.Iterable):
        return type(inputs)(
            cast_tensor_type(item, src_type, dst_type) for item in inputs)
    else:
        return inputs


def wrap_auto_fp16(apply_to=None, out_fp32=False):
    """Decorator to enable fp16 training automatically.

    This decorator is useful when you write custom modules and want to support
    mixed precision training. If inputs arguments are fp32 tensors, they will
    be converted to fp16 automatically. Arguments other than fp32 tensors are
    ignored.

    Args:
        apply_to (Iterable, optional): The argument names to be converted.
            `None` indicates all arguments.
        out_fp32 (bool): Whether to convert the output back to fp32.

    Example:

        >>> import torch.nn as nn
        >>> class MyModule1(nn.Module):
        >>>
        >>>     # Convert x and y to fp16
        >>>     @auto_fp16()
        >>>     def forward(self, x, y):
        >>>         pass

        >>> import torch.nn as nn
        >>> class MyModule2(nn.Module):
        >>>
        >>>     # convert pred to fp16
        >>>     @auto_fp16(apply_to=('pred', ))
        >>>     def do_something(self, pred, others):
        >>>         pass
    """

    def auto_fp16_wrapper(old_func):

        @functools.wraps(old_func)
        def new_func(*args, **kwargs):
            # check if the module has set the attribute `fp16_enabled`, if not,
            # just fallback to the original method.
            if not isinstance(args[0], torch.nn.Module):
                raise TypeError('@auto_fp16 can only be used to decorate the '
                                'method of nn.Module')
            if not (hasattr(args[0], 'fp16_enabled') and args[0].fp16_enabled):
                return old_func(*args, **kwargs)
            # get the arg spec of the decorated method
            args_info = getfullargspec(old_func)
            # get the argument names to be casted
            args_to_cast = args_info.args if apply_to is None else apply_to
            # convert the args that need to be processed
            new_args = []
            # NOTE: default args are not taken into consideration
            if args:
                arg_names = args_info.args[:len(args)]
                for i, arg_name in enumerate(arg_names):
                    if arg_name in args_to_cast:
                        new_args.append(
                            cast_tensor_type(args[i], torch.float, torch.half))
                    else:
                        new_args.append(args[i])
            # convert the kwargs that need to be processed
            new_kwargs = {}
            if kwargs:
                for arg_name, arg_value in kwargs.items():
                    if arg_name in args_to_cast:
                        new_kwargs[arg_name] = cast_tensor_type(
                            arg_value, torch.float, torch.half)
                    else:
                        new_kwargs[arg_name] = arg_value
            # apply converted arguments to the decorated method
            output = old_func(*new_args, **new_kwargs)
            # cast the results back to fp32 if necessary
            if out_fp32:
                output = cast_tensor_type(output, torch.half, torch.float)
            return output

        return new_func
    return auto_fp16_wrapper


def convert_tensor(dat, dst_type=torch.float):
    if isinstance(dat, torch.Tensor):
        if dat.dtype != dst_type:
            return dat.to(dst_type)
        return dat
    elif isinstance(dat, abc.Mapping):
        ndict = {}
        for k, v in dat.items():
            if v.dtype != dst_type:
                v = v.to(dst_type)
            ndict[k] = v
        return ndict
    elif isinstance(dat, abc.Iterable):
        return type(dat)(item.to(dst_type) for item in dat)
    else:
        raise ValueError(f"Unsupported type in auto fp16: {type(dat)}")
def convert_fp32(dat):
    return convert_tensor(dat, torch.float)
def convert_fp16(dat):
    return convert_tensor(dat, torch.half)

def auto_fp16(fp16_state, dat,):
    if fp16_state != 1:
        return dat
    return convert_tensor(dat, torch.half)

def force_fp32(apply_to=None, out_fp16=False):
    """Decorator to convert input arguments to fp32 in force.

    This decorator is useful when you write custom modules and want to support
    mixed precision training. If there are some inputs that must be processed
    in fp32 mode, then this decorator can handle it. If inputs arguments are
    fp16 tensors, they will be converted to fp32 automatically. Arguments other
    than fp16 tensors are ignored.

    Args:
        apply_to (Iterable, optional): The argument names to be converted.
            `None` indicates all arguments.
        out_fp16 (bool): Whether to convert the output back to fp16.

    Example:

        >>> import torch.nn as nn
        >>> class MyModule1(nn.Module):
        >>>
        >>>     # Convert x and y to fp32
        >>>     @force_fp32()
        >>>     def loss(self, x, y):
        >>>         pass

        >>> import torch.nn as nn
        >>> class MyModule2(nn.Module):
        >>>
        >>>     # convert pred to fp32
        >>>     @force_fp32(apply_to=('pred', ))
        >>>     def post_process(self, pred, others):
        >>>         pass
    """

    def force_fp32_wrapper(old_func):

        @functools.wraps(old_func)
        def new_func(*args, **kwargs):
            # check if the module has set the attribute `fp16_enabled`, if not,
            # just fallback to the original method.
            if not isinstance(args[0], torch.nn.Module):
                raise TypeError('@force_fp32 can only be used to decorate the '
                                'method of nn.Module')
            if not (hasattr(args[0], 'fp16_enabled') and args[0].fp16_enabled):
                return old_func(*args, **kwargs)
            # get the arg spec of the decorated method
            args_info = getfullargspec(old_func)
            # get the argument names to be casted
            args_to_cast = args_info.args if apply_to is None else apply_to
            # convert the args that need to be processed
            new_args = []
            if args:
                arg_names = args_info.args[:len(args)]
                for i, arg_name in enumerate(arg_names):
                    if arg_name in args_to_cast:
                        new_args.append(
                            cast_tensor_type(args[i], torch.half, torch.float))
                    else:
                        new_args.append(args[i])
            # convert the kwargs that need to be processed
            new_kwargs = dict()
            if kwargs:
                for arg_name, arg_value in kwargs.items():
                    if arg_name in args_to_cast:
                        new_kwargs[arg_name] = cast_tensor_type(
                            arg_value, torch.half, torch.float)
                    else:
                        new_kwargs[arg_name] = arg_value
            # apply converted arguments to the decorated method
            output = old_func(*new_args, **new_kwargs)
            # cast the results back to fp32 if necessary
            if out_fp16:
                output = cast_tensor_type(output, torch.float, torch.half)
            return output

        return new_func

    return force_fp32_wrapper


# def allreduce_grads(params, coalesce=True, bucket_size_mb=-1):
#     _allreduce_grads(params, coalesce=coalesce, bucket_size_mb=bucket_size_mb)


def wrap_fp16_model(model: "nn.Module", skip_modules=None):
    """Wrap the FP32 model to FP16.
    fp16_state: -1 disable fp16 mode, 
                 0 fp16 not enabled, 
                 1 fp16 enabled, 
    1. Convert FP32 model to FP16.
    2. Remain some necessary layers to be FP32, e.g., normalization layers.

    Args:
        model (nn.Module): Model in FP32.
    """
    model.half()  ## half the whole model

    for name, m in model.named_modules():
        if skip_modules is not None:
            skip_it = False
            for sm in skip_modules:
                if sm in name:
                    skip_it = True
                    break
            if skip_it:
                continue
        # set `fp16_enabled` flag
        if not hasattr(m, 'fp16_state'):
            ### if no fp16_state
            # m.float()
            # skip_fp16_models.append(name)
            continue
        if m.fp16_state < 0:
            ### fp16 is disabled
            m.float()
        elif m.fp16_state == 0:
            # children.append(m)
            m.half()
            # patch the normalization layers to make it work in fp32 mode
            patch_norm_fp32(m)
            m.fp16_state = 1
        elif m.fp16_state == 1:
            ## do nothing since it has been fp16
            pass
        else:
            raise ValueError(f"Module fp16_state error {name}")
        
    # print(f"Following modules not support fp16 mdoe: {skip_fp16_models}\n")
def wrap_fp32_model(model: "nn.Module"):
    """Wrap the FP32 model to FP16.
    fp16_state: -1 disable fp16 mode, 
                 0 fp16 not enabled, 
                 1 fp16 enabled, 
    1. Convert FP32 model to FP16.
    2. Remain some necessary layers to be FP32, e.g., normalization layers.

    Args:
        model (nn.Module): Model in FP32.
    """
    model.float()  ## half the whole model

    for name, m in model.named_modules():
        if not hasattr(m, 'fp16_state'):
            ### if no fp16_state
            # m.float()
            # skip_fp16_models.append(name)
            continue
        if m.fp16_state < 0:
            ### fp16 is disabled
            m.float()
        elif m.fp16_state == 0:
            ## do nothing since it has been fp32
            pass
        elif m.fp16_state == 1:
            m.fp16_state = 0
        else:
            raise ValueError(f"Module fp16_state error {name}")
        
def patch_norm_fp32(module):
    """Recursively convert normalization layers from FP16 to FP32.

    Args:
        module (nn.Module): The modules to be converted in FP16.

    Returns:
        nn.Module: The converted module, the normalization layers have been
            converted to FP32.
    """
    if isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.GroupNorm)):
        module.float()
        if isinstance(module, nn.GroupNorm) or torch.__version__ < '1.3':
            module.forward = patch_forward_method(module.forward, torch.half,
                                                  torch.float)
    for child in module.children():
        patch_norm_fp32(child)
    return module


def patch_forward_method(func, src_type, dst_type, convert_output=True):
    """Patch the forward method of a module.

    Args:
        func (callable): The original forward method.
        src_type (torch.dtype): Type of input arguments to be converted from.
        dst_type (torch.dtype): Type of input arguments to be converted to.
        convert_output (bool): Whether to convert the output back to src_type.

    Returns:
        callable: The patched forward method.
    """

    def new_forward(*args, **kwargs):
        output = func(*cast_tensor_type(args, src_type, dst_type),
                      **cast_tensor_type(kwargs, src_type, dst_type))
        if convert_output:
            output = cast_tensor_type(output, dst_type, src_type)
        return output

    return new_forward

from torch.cuda.amp import autocast

# Decorator to disable autocast
def autocast_disabled(func):
    def wrapper(*args, **kwargs):
        with autocast(enabled=False):
            return func(*args, **kwargs)
    return wrapper
