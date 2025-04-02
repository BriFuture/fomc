_base_ = ["ds_fomc_shot.py", "rt_shot.py", "../model_fomc_shot.py"]

_split=1
_shot=30
_seed=5

experiment = dict(
    work_dir=f"checkpoints/dior_r/fomc/split{_split}/shot{_shot}",
)

_data_root = f"datasets/DIOR-R/split/mask_seed{_seed}_shot{_shot}/"


train = dict(
    solver = dict(
        decay_epochs = [160,],
        total_epochs = 180,
        learning_rate = 0.003,
    ),
    eval_checkpoints=5,
)

dataset = dict(

    train = dict(
        shot=_shot,
        ann_file= _data_root + "ImageSets/Main/train.txt",
        img_dir = _data_root,
        img_subdir = "JPEGImages",
    ),
)

_odm_head = dict(
    cfg=dict(
        ignore_unused_object_loss = True,
        contrast = dict(
            decay_steps=[4000, ],
            decay_rate = 0.3,
            contrast_weight_step=100,
            contrast_weight = 0.1,
        ),
    ),

)

model = dict(
    odm_head = _odm_head,
)