_base_ = ["ds_fomc.py", "../rt_dior.py", "../model_fomc_base.py"]

_split=1

experiment = dict(
    work_dir=f"checkpoints/dior_r/fomc/split{_split}/base",
)
train = dict(
    solver = dict(
        learning_rate = 0.01,
        decay_epochs = [ 16, 22],
        total_epochs = 24,
    ),
)