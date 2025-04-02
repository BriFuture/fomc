import torch
import torch.nn as nn

def calculate_state_dict_size(model: "nn.Module"):
    state_dict = model.state_dict()
    param_size_in_bytes = 0
    for name, tensor in state_dict.items():
        element_size = tensor.element_size() 
        total_elements = tensor.numel() 
        param_size_in_bytes += element_size * total_elements
    # total_params = sum(param.numel() for param in state_dict.values())
    # 以 float32 为例，每个参数占 4 字节
    # param_size_in_bytes = total_params * 4
    param_size_in_MB = param_size_in_bytes / (1024 ** 2)
    return param_size_in_MB
