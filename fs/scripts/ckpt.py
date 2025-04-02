
"""
1.1 support fp16
1.2 class hierachecy
1.3 add surgery, fix softmax problem
"""
__version__ = "1.3"

from copy import deepcopy
from mmcv.cnn.utils.weight_init import bias_init_with_prob
import sys
import torch
import os
import argparse
def construct_parser(param_names=None):
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument('--src1', type=str, default='',
                        help='Path to the main checkpoint')
    parser.add_argument('--src2', type=str, default='',
                        help='Path to the secondary checkpoint (for combining)')
    parser.add_argument('--save-dir', type=str, default='',
                        help='Save directory')

    # Surgery method
    parser.add_argument('--method', choices=['combine', 'remove', 'randinit', "compare"],
                        required=True,
                        help='Surgery method. combine = '
                             'combine checkpoints. remove = for fine-tuning on '
                             'novel dataset, remove the final layer of the '
                             'base detector. randinit = randomly initialize '
                             'novel weights.')
    # Targets
    parser.add_argument('--param-name', default=param_names,
                        help='Target parameter names')
    parser.add_argument('--tar-name', type=str, default='model_reset',
                        help='Name of the new ckpt')
    parser.add_argument('--num-class', type=int, default=15,
                        help='class number')
    parser.add_argument('--softmax', action="store_true",
                        help='Use softmax')
    parser.add_argument('--fp16', action="store_true", 
                        help='enable fp16')
    parser.add_argument('--reserve-base-branch', action="store_true", 
                        help='copy base branch data into novel, for original version only')
    parser.add_argument('--prob_bias', type=int, default=1,)
    parser.add_argument('--bbox_anchors', type=int, default=1,)

    return parser

def save_ckpt(ckpt, save_name):
    torch.save(ckpt, save_name)
    print('=============\nsave changed ckpt to {}'.format(save_name))


def reset_ckpt(ckpt):
    if 'scheduler' in ckpt:
        del ckpt['scheduler']
    if 'optimizer' in ckpt:
        del ckpt['optimizer']
    if 'iteration' in ckpt:
        ckpt['iteration'] = 0
    if 'epoch' in ckpt:
        ckpt['epoch'] = 0
    if 'meta' not in ckpt:
        return
    m = ckpt['meta']
    if 'CLASSES' in m:
        m.pop("CLASSES")

    if 'epoch' in m:
        m['epoch'] = 0

    if 'iter' in m:
        m['iter'] = 0
        
class Surgery() :
    """
    class mapping tuple, for foreground only
    """
    def __init__(self, param_names, args, class_mapping=None):
        num_class = args['num_class']
        if args["softmax"]: 
            num_class += 1
        self.num_class = num_class
        self.param_names = param_names
        self.args = args
        self.fp16 = args["fp16"]
        self.softmax = args["softmax"]
        self.prob_bias = args['softmax'] and args['prob_bias'] > 0
        
        self.class_mapping = class_mapping
        
        print("[Surgery] ", args)
        print("[Surgery] ", param_names, num_class, "class mapping", class_mapping)

    def start(self, method):
        if method == 'combine':
            self.combine()
        elif method == "compare":
            self.compare()
        else:
            self.surgery()
    def _surgery(self, param_name, ckpt):
        """
        Either remove the final layer weights for fine-tuning on novel dataset or
        append randomly initialized weights for the novel classes.

        Note: The base detector for LVIS contains weights for all classes, but only
        the weights corresponding to base classes are updated during base training
        (this design choice has no particular reason). Thus, the random
        initialization step is not really necessary.
        """
        print("dev: ",  param_name)
        state_dict = ckpt['state_dict']
        for is_weight in (True, False):
            tar_size = self.num_class
            weight_name = param_name + ('.weight' if is_weight else '.bias')
            if weight_name not in state_dict:
                continue
            pretrained_weight = state_dict[weight_name]
            prev_cls = pretrained_weight.size(0)
            if self.is_bg_affect(param_name) :
                prev_cls -= 1
            if 'bbox_head' in param_name:
                tar_size = tar_size * self.args['bbox_anchors']
            print(f"[{param_name}] Pretrained weight shape", pretrained_weight.shape, prev_cls)
            feat_size = pretrained_weight.shape[1:]
            if is_weight:
                new_weight = pretrained_weight.new_zeros((tar_size, *feat_size))
                torch.nn.init.constant_(new_weight, -0.01)
            else:
                bias = bias_init_with_prob(0.01) if not self.prob_bias else 0.0
                new_weight = pretrained_weight.new_zeros(tar_size) + bias
            new_weight[:prev_cls] = pretrained_weight[:prev_cls]

            if self.is_bg_affect(param_name) :
                new_weight[-1] = pretrained_weight[-1]  # bg class
            
            if not is_weight:
                print(f"{param_name} New bias: {new_weight}")
            else:
                d = [f"{x.mean().item():.5f}" for x in new_weight]
                print(f"{param_name} New weight: [{', '.join(d)}]")

            state_dict[weight_name] = new_weight.to(pretrained_weight.dtype)
    
    def surgery(self):
        """
        Either remove the final layer weights for fine-tuning on novel dataset or
        append randomly initialized weights for the novel classes.

        Note: The base detector for LVIS contains weights for all classes, but only
        the weights corresponding to base classes are updated during base training
        (this design choice has no particular reason). Thus, the random
        initialization step is not really necessary.
        """
        args = self.args

        ckpt = torch.load(args["src1"])
        reset_ckpt(ckpt)
        save_path = self.get_pth_save_path(args)
        # Surgery
        param_names = self.param_names
        state_dict = ckpt['state_dict']
        keys = list(state_dict.keys())
        i = 0
        ## 保留base 分支
        if args["reserve_base_branch"]:
            for k in keys:
                if k.startswith("bbox_head"):
                    state_dict[f"base_{k}"] = state_dict[k]
                    i += 1
                    # state_dict[f"{nk}.bias"] = state_dict[f"{k}.bias"]
            print(">> Param changed ", i)
            print("===============")
        for param_name in param_names:
            # print(param_name)
            self._surgery(param_name, ckpt)
        save_ckpt(ckpt, save_path)
        # self.surgery_loop(surgery)
    def is_bg_affect(self, name):
        return 'cls' in name and self.softmax
    
    def get_pth_save_path(self, args):
        
        if args["method"] == 'combine':
            save_name = args["tar_name"] + '_combine.pth'
        else:
            save_name = args["tar_name"] + '_' + \
                ('remove' if args["method"] == 'remove' else 'surgery') + '.pth'
        if args["save_dir"] == '':
            # By default, save to directory of src1
            save_dir = os.path.dirname(args["src1"])
        else:
            save_dir = args["save_dir"]
        save_path = os.path.join(save_dir, save_name)
        os.makedirs(save_dir, exist_ok=True)
        return save_path
    
    def combine(self):
        """
        Combine base detector with novel detector. Feature extractor weights are
        from the base detector. Only the final layer weights are combined.
        """
        args = self.args

        ckpt1 = torch.load(args["src1"])
        ckpt2 = torch.load(args["src2"])
        reset_ckpt(ckpt1)
        save_path = self.get_pth_save_path(args)
        # Surgery
        param_names = self.param_names
        i = 0
        print("Param changed ", i)
        print("===============")
        tar_sizes = [self.num_class, self.num_class]
        if args['skip_combine']:
            self._skip_combine(param_names, ckpt1, ckpt2)
        else:
          for idx, (param_name, tar_size) in enumerate(zip(param_names, tar_sizes)):
            # print(param_name)
            self._combine(param_name, tar_size, ckpt1, ckpt2)
        save_ckpt(ckpt1, save_path)

    def _skip_combine(self, param_names: list, ckpt, ckpt2):
        print("Combine skip: ",  param_names)
        sd1: dict = ckpt['state_dict']
        sd2 = ckpt2['state_dict']
        for name, parameter in sd1.items():
            hit = False
            for pn in param_names:
                if pn in name:
                    hit = True
            if hit: continue
            sd1[name] = sd2[name]
            diff = (sd2[name] - parameter).to(torch.float).mean().item()
            print(name, f"{diff:.2f}")

    def _combine(self, param_name: str, tar_size: int, ckpt, ckpt2):
        print("Combine: ",  param_name)
        for is_weight in (True, False):
            if not is_weight and param_name + '.bias' not in ckpt['state_dict']:
                return
            weight_name = param_name + ('.weight' if is_weight else '.bias')
            pretrained_weight = ckpt['state_dict'][weight_name]
            prev_cls = pretrained_weight.size(0)
            feat_size = pretrained_weight.shape[1:]
            if is_weight:
                # feat_size = pretrained_weight.size(1)
                new_weight = torch.rand((tar_size, *feat_size))
            else:
                new_weight = pretrained_weight.new_zeros((tar_size, ))
            new_weight[:prev_cls] = pretrained_weight[:prev_cls]
            ckpt2_weight = ckpt2['state_dict'][weight_name]
            prev_cls = ckpt2_weight.size(0)
            print(ckpt2_weight.shape, new_weight.shape)
            
            if 'cls_score' in param_name:
                new_weight[prev_cls:-1] = ckpt2_weight[:-1]
                new_weight[-1] = pretrained_weight[-1]
            else:
                new_weight[:prev_cls] = ckpt2_weight
            ckpt['state_dict'][weight_name] = new_weight

    

    def compare(self):
        args = self.args
        def surgery(param_names, tar_size, ckpt):
            print("dev: ",  param_names)
            param_name = param_names[0]
            state_dict = ckpt['state_dict']
            for is_weight in (True, False):
                weight_name = param_name + ('.weight' if is_weight else '.bias')
                new_weight_name = param_names[1] + ('.weight' if is_weight else '.bias')
                pretrained_weight = state_dict[weight_name]
                new_pretrained_weight = state_dict[new_weight_name]

                prev_cls = pretrained_weight.size(0)
                print(torch.mean(pretrained_weight - new_pretrained_weight[:prev_cls]))

        ckpt = torch.load(args["src1"])

        # Surgery
        param_names = self.param_names
        tar_sizes = [self.num_class, self.num_class]
        for idx, (param_name, tar_size) in enumerate(zip(param_names, tar_sizes)):
            # print(param_name)
            surgery(param_name, tar_size, ckpt)
    

