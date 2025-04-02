# coding=utf-8

from gdet.train_and_eval import single_main as main
import sys

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 0:
        args = [
            # "configs/dota/orcnn/split1/ex_shot30.py",
            # "--load_from=checkpoints/dota/orcnn/split1/mm_shot30/val_best_ckpt.pth",
            # "configs/dota/s2anet/split1/ex_shot30.py",
            # "--load_from=checkpoints/dota/s2anet/split1/shot30/val_best_ckpt.pth",
            # "configs/dota/r3det/split1/ex_shot30.py",
            # "--load_from=checkpoints/dota/r3det/split1/shot30/val_best_ckpt.pth",

            "configs/dota/fomc/split1/ex_shot30.py",
            "--load_from=checkpoints/dota/fomc/split1/shot30/val_best_ckpt.pth",
            ###
            "--no_eval",
            "--calc_flops",
        ]

    main(args)
