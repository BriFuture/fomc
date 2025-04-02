# coding=utf-8

from gdet.train_and_eval import main
import sys

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 0:
        args = [
            # "configs/hrsc/orcnn/split1/ex_base.py",
            # "configs/hrsc/orcnn/ex_orcnn.py",
            "configs/hrsc/fomc/split1/ex_base_coco.py",
            "--load_from=checkpoints/weights/hrsc/fomc/split1/base/val_best_ckpt.pth",
            # "--train",
            # "--no_save_best",
        ]

    main(args)
