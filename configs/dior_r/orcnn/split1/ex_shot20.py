_base_ = ["ds_orcnn_shot.py", "rt_shot.py", "../model_orcnn_shot.py"]

_split=1
_shot=20
_seed=5

experiment = dict(
    work_dir=f"checkpoints/dior_r/orcnn/split{_split}/shot{_shot}",
)

_data_root = f"datasets/DIOR-R/split/seed{_seed}_shot{_shot}/"

train = dict(
    solver = dict(
        decay_epochs = [90,],
        total_epochs = 100,
    ),
    eval_checkpoints=5,
)

dataset = dict(

    train = dict(
        shot=_shot,
        ann_file= _data_root+"ImageSets/Main/train.txt",
        img_dir = _data_root,
        img_subdir = "JPEGImages",
    ),
)