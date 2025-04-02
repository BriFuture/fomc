# coding=utf-8

from gdet.train_and_eval import main
import sys

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 0:
        args = [
            "configs/dior_r/s2anet/split1/ex_shot3.py",
            "--load_from=checkpoints/dior_r/s2anet/split1/shot5/val_best_ckpt.pth",
            # "--train",
            # "--no_save_best",
        ]

    main(args)
