_base_ = ["../ds_dota_origin.py",]

_split=1
_shot=10
_seed=5


_data_root = f"datasets/DOTA_v1/train_ss/split/mask_seed{_seed}_shot{_shot}/"

dataset = dict(

    train = dict(
        shot = _shot,
        ann_file= _data_root+"ImageSets/Main/train.txt",
        img_dir = _data_root + "images",
        ann_folder = _data_root+"annfiles",
    ),
)

