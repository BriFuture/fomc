_base_ = ["ds_orcnn_shot.py", "rt_shot.py", "../model_orcnn_shot_mcl.py"]

_split=2
_shot=20
_seed=5

experiment = dict(
    work_dir=f"checkpoints/dota/orcnn/split{_split}/mm_shot{_shot}",
)

_data_root = f"datasets/DOTA_v1/train_ss/split/mask_seed{_seed}_shot{_shot}/"


train = dict(
    solver = dict(
        decay_epochs = [180,],
        total_epochs = 200,
        learning_rate = 0.003,
    ),
    eval_checkpoints=5,
    log_interval = 30,
)

dataset = dict(

    train = dict(
        shot = _shot,
        ann_file=_data_root + "ImageSets/train.txt",
        img_dir = _data_root + "images",
        ann_folder = _data_root+"annfiles",
    ),
)
