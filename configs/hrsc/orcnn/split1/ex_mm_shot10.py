_base_ = ["ds_orcnn_shot.py", "rt_shot.py", "../model_orcnn_shot_mcl.py"]

_split=1
_shot=10
_seed=5

experiment = dict(
    work_dir=f"checkpoints/hrsc/orcnn/split{_split}/mm_shot{_shot}",
)

_data_root = f"datasets/HRSC2016/FullDataSet/split/mask_seed{_seed}_shot{_shot}/"

train = dict(
    solver = dict(
        decay_epochs = [140,],
        total_epochs = 150,
        learning_rate = 0.005,
    ),
    eval_checkpoints=5,
)

dataset = dict(

    train = dict(
        shot=_shot,
        ann_file= _data_root + "ImageSets/train.txt",
        img_dir = _data_root ,
    ),
)

_roi_head = dict(
    cfg = dict(
        contrast = dict(
            contrast_weight = 0.03,
        ),
    )
)

model = dict(
    roi_head=_roi_head,
)