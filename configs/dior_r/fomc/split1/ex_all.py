_base_ = ["ds_fomc_shot.py", "rt_shot.py", "../model_fomc_all.py"]

_split=1

experiment = dict(
    work_dir=f"checkpoints/dior_r/fomc/split{_split}/all",
    load_from="",
)
train = dict(
    solver = dict(
        learning_rate = 0.01,
        decay_epochs = [ 16, 22],
        total_epochs = 24,
        warmup_steps = 500,
    ),
)

_odm_head = dict(
    cfg = dict(
        contrast = dict(
            decay_steps=[4000, ],
            decay_rate = 0.2,
            contrast_weight_step=100,
            contrast_weight = 0.01,
        ),

    )
)

model = dict(
    odm_head = _odm_head,
)