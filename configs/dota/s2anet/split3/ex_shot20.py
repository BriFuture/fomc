_base_ = ["ds_s2anet_shot.py", "rt_shot.py", "../model_s2anet_shot.py"]

_split=3
_shot=20
_seed=5

experiment = dict(
    work_dir=f"checkpoints/dota/s2anet/split{_split}/shot{_shot}",
)

_data_root = f"datasets/DOTA_v1/train_ss/split/seed{_seed}_shot{_shot}/"


train = dict(
    solver = dict(
        decay_epochs = [190,],
        total_epochs = 200,
        learning_rate = 0.0025,
    ),
    log_interval = 30,
    eval_checkpoints=20,
)


dataset = dict(

    train = dict(
        shot = _shot,
        ann_file= _data_root+"ImageSets/Main/train.txt",
        img_dir = _data_root + "images",
        ann_folder = _data_root+"annfiles",
    ),
)