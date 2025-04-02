_base_ = ["ds_r3det_shot.py", "rt_shot.py", "../model_r3det_shot.py"]

_split=1
_shot=10
_seed=5

experiment = dict(
    work_dir=f"checkpoints/dior_r/r3det/split{_split}/shot{_shot}",
)

_data_root = f"datasets/DIOR-R/split/seed{_seed}_shot{_shot}/"

train = dict(
    solver = dict(
        decay_epochs = [380,],
        total_epochs = 300,
        learning_rate = 0.002,
    ),
    eval_checkpoints=20,
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