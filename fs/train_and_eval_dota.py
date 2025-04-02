# coding=utf-8

from gdet.train_and_eval import main
import sys

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 0:
        args = [
            # "configs/dota/ex_s2anet.py",
            # "--load_from=weights/s2anet_r50_fpn_1x_dota_le135-5dfcf396.pth",
            # "--load_from=checkpoints/dota/s2anet/all/train_ckpt_ep8.pth",
            # "configs/dota/orcnn/ex_orcnn.py",
            # "--load_from=checkpoints/dota/orcnn/all/val_best_ckpt.pth",
            # "configs/dota/fomc/split1/ex_shot30_ab_mcl.py",
            "configs/dota/orcnn/split1/ex_shot30_mask_mcl.py",
            "--train",
            # "--load_from=/media/fut/sda4/s2anet/work_dirs/fomc_dota/base4/softmax/model_reset_surgery.pth",
        ]

    main(args)
