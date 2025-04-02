_base_ = ["ds_r3det_shot.py",  "rt_shot.py", "../model_r3det_shot.py"]

_split=1
_shot=10
_seed=5

experiment = dict(
    work_dir=f"checkpoints/dota/r3det/split{_split}/shot{_shot}",
)

_data_root = f"datasets/DOTA_v1/train_ss/split/seed{_seed}_shot{_shot}/"
train = dict(
    solver = dict(
        decay_epochs = [390,],
        total_epochs = 400,
        learning_rate = 0.002,
    ),
    log_interval = 20,
    eval_checkpoints=10,
)

dataset = dict(

    train = dict(
        shot = _shot,
        ann_file= _data_root+"ImageSets/Main/train.txt",
        img_dir = _data_root + "images",
        ann_folder = _data_root+"annfiles",
    ),
)