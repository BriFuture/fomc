_base_ = ["ds_fomc_shot.py", "rt_shot.py", "../model_fomc_shot.py"]

_split=1
_shot=100
_seed=5

experiment = dict(
    work_dir=f"checkpoints/dota/fomc/split{_split}/shot{_shot}",
)

_data_root = f"datasets/DOTA_v1/train_ss/split/mask_seed{_seed}_shot{_shot}/"

train = dict(
    log_interval = 20,
    solver = dict(
        decay_epochs = [90,],
        total_epochs = 120,
        learning_rate = 0.005,
    ),
    eval_checkpoints=3,
)


dataset = dict(

    train = dict(
        shot = _shot,
        ann_file= _data_root + "ImageSets/Main/train.txt",
        img_dir = _data_root + "images",
        ann_folder = _data_root+"annfiles",
    ),
)

_odm_head = dict(
    cfg = dict(
        contrast = dict(
            decay_steps=[6000, ],
            decay_rate = 0.2,
            contrast_weight_step=100,
            contrast_weight = 0.2,
        ),

    )
)

model = dict(
    freeze_neck=False,
    neck=dict(
        type="FsFPN",
        frozen=[],
    ),
    odm_head = _odm_head,
)