_base_ = ["ds_fomc.py", "../rt_fomc.py", "../model_fomc_base.py"]

_split=1

experiment = dict(
    work_dir=f"checkpoints/hrsc/fomc/split{_split}/base",
)

train = dict(
    solver = dict(
        learning_rate = 0.01,
        decay_epochs = [ 24, 33],
        total_epochs = 36,
    ),
    eval_checkpoints=5,
)