import torch
from fs.scripts.ckpt import construct_parser, Surgery
import sys

def main():
    args = sys.argv[1:]
    if len(args) == 0:
        args = [
            "--method", "randinit", 
        ]
        args.extend([
            "--src1", "checkpoints/weights/dior_r/s2anet/split1/base/val_best_ckpt.pth",
            "--save-dir", "checkpoints/weights/dior_r/s2anet/split1/sigmoid",
            "--num-class=20",
        ])
    param_name = [
        'bbox_head.fam_cls',
        'fam_head.retina_cls',
        'odm_head.odm_cls',
        # 'bbox_head.fam_reg',
        'bbox_head.odm_cls',
        # 'bbox_head.odm_reg'
        ### r3det
        'bbox_head.retina_cls',
        'refine_head.0.retina_cls',
        'refine_head.1.retina_cls',
    ]
    parser = construct_parser(param_name)
  
    args = vars(parser.parse_args(args))

    s = Surgery(param_name, args)
    s.start(args["method"])


if __name__ == '__main__':
    main()