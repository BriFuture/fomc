_base_ = ["ds_fomc_shot.py", "../rt_dota.py", "../model_fomc_all.py"]

experiment = dict(
    work_dir="checkpoints/dota/fomc/split1/all",
)

train = dict(
    solver = dict(
        learning_rate = 0.001,
        decay_epochs = [ 12, 16],
        total_epochs = 18,
    ),
)

_odm_head = dict(
    cfg=dict(
        contrast = dict(
            decay_steps=[4000, ],
            decay_rate = 0.3,
            contrast_weight_step=100,
            contrast_weight = 0.0,
        ),
    ),
)

model = dict(
    odm_head = _odm_head,
)