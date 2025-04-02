_base_ = ["ds_fomc_shot.py", "rt_shot.py", "../model_fomc_all.py"]

_split=1

experiment = dict(
    work_dir=f"checkpoints/hrsc/fomc/split{_split}/all",
    load_from="",
)

train = dict(
    solver = dict(
        learning_rate = 0.01,
        decay_epochs = [ 24, 33],
        total_epochs = 36,
    ),
    eval_checkpoints=5,
)

_odm_head = dict(
    cfg=dict(
        ignore_unused_object_loss = True,
        contrast = dict(
            decay_steps=[4000, ],
            decay_rate = 0.3,
            contrast_weight_step=100,
            contrast_weight = 0.005,
        ),
    ),
)

model = dict(
    odm_head = _odm_head,
)
