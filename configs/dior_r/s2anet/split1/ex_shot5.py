_base_ = ["ds_s2anet_shot.py", "rt_shot.py", "../model_s2anet_shot.py"]

_split=1
_shot=5
_seed=5

experiment = dict(
    work_dir=f"checkpoints/dior_r/s2anet/split{_split}/shot{_shot}",
)

_data_root = f"datasets/DIOR-R/split/seed{_seed}_shot{_shot}/"

train = dict(
    solver = dict(
        decay_epochs = [220,],
        total_epochs = 240,
        learning_rate = 0.0015,
    ),
    eval_checkpoints = 40,
    log_interval = 5,
)

dataset = dict(

    train = dict(
        shot=_shot,
        ann_file= _data_root+"ImageSets/Main/train.txt",
        img_dir = _data_root,
        img_subdir = "JPEGImages",
    ),
)