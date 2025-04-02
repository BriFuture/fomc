_base_ = ["ds_fomc.py", "../rt_dota.py", "../model_fomc_base.py"]

_split=1

experiment = dict(
    work_dir=f"checkpoints/dota/fomc/split{_split}/base",
)

train = dict(
    solver = dict(
        learning_rate = 0.01,
        decay_epochs = [ 12, 16],
        total_epochs = 18,
        warmup_steps = 500,
    ),
)