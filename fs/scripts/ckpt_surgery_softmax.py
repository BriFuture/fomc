import torch
from fs.scripts.ckpt import construct_parser, Surgery
import sys
if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 0:
        args = [
            "--method", "randinit", 
        ]
        ### DIOR
        # args.extend([
        #     "--src1", "checkpoints/weights/dior_r/orcnn/split1/base/val_best_ckpt.pth",
        #     "--softmax",
        #     "--save-dir", "checkpoints/weights/dior_r/orcnn/split1/softmax",
        #     "--num-class=20",
        # ])
        args.extend([
            "--src1", "checkpoints/weights/dior_r/fomc/split1/base/val_best_ckpt.pth",
            "--softmax",
            "--save-dir", "checkpoints/weights/dior_r/fomc/split1/softmax",
            "--num-class=20", 
            # "--prob_bias=0"
        ])    
        ### HRSC    
        args.extend([
            "--src1", "checkpoints/weights/hrsc/orcnn/split1/base/val_best_ckpt.pth",
            "--softmax",
            "--save-dir", "checkpoints/weights/hrsc/orcnn/split1/softmax",
            "--num-class=20",
        ])
    param_name = [
        'bbox_head.fam_cls',
        # 'bbox_head.fam_reg',
        'bbox_head.odm_cls',
        # 'bbox_head.odm_reg'
        "roi_head.bbox_head.fc_cls",
        'fam_head.retina_cls',
        'odm_head.odm_cls',
    ]
    parser = construct_parser(param_name)
  
    args = vars(parser.parse_args(args))

    s = Surgery(param_name, args)
    s.start(args["method"])
