_base_ = ["ds_orcnn_shot.py", "rt_shot.py", "../model_orcnn_shot.py"]

_split=3
_shot=5
_seed=5

experiment = dict(
    work_dir=f"checkpoints/dota/orcnn/split{_split}/shot{_shot}",
)

_data_root = f"datasets/DOTA_v1/train_ss/split/seed{_seed}_shot{_shot}/"

train = dict(
    solver = dict(
        decay_epochs = [290,],
        total_epochs = 400,
        learning_rate = 0.0015,
    ),
    eval_checkpoints=40,
    log_interval = 10,
)

dataset = dict(

    train = dict(
        shot = _shot,
        ann_file= _data_root+"ImageSets/Main/train.txt",
        img_dir = _data_root + "images",
        ann_folder = _data_root+"annfiles",
    ),
)