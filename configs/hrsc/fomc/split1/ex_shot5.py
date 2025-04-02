_base_ = ["ds_fomc_shot.py", "rt_shot.py", "../model_fomc_shot.py"]

_split=1
_shot=5
_seed=5

experiment = dict(
    work_dir=f"checkpoints/hrsc/fomc/split{_split}/shot{_shot}",
)

_data_root = f"datasets/HRSC2016/FullDataSet/split/mask_seed{_seed}_shot{_shot}/"

train = dict(
    solver = dict(
        decay_epochs = [140,],
        total_epochs = 150,
        learning_rate = 0.0025,
    ),
    eval_checkpoints=10,
)

dataset = dict(

    train = dict(
        shot = _shot,
        ann_file=_data_root + "ImageSets/train.txt",
        img_dir = _data_root ,
    ),
)