_base_ = ["ds_r3det_shot.py", "rt_shot.py", "../model_r3det_shot.py"]

_split=1
_shot=10
_seed=5

experiment = dict(
    work_dir=f"checkpoints/hrsc/r3det/split{_split}/shot{_shot}",
)

_data_root = f"datasets/HRSC2016/FullDataSet/split/seed{_seed}_shot{_shot}/"


train = dict(
    solver = dict(
        decay_epochs = [ 110,],
        total_epochs = 120,
        learning_rate = 0.003,
    ),
    eval_checkpoints=5,
)

dataset = dict(

    train = dict(
        shot = _shot,
        ann_file= _data_root + "ImageSets/train.txt",
        img_dir = _data_root,
    ),
)