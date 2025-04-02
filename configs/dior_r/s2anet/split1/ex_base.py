_base_ = ["ds_s2anet.py", "../rt_dior.py", "../model_s2anet_base.py"]

_split=1

experiment = dict(
    work_dir=f"checkpoints/dior_r/s2anet/split{_split}/base",
)

train = dict(
    solver = dict(
        learning_rate = 0.01,
        decay_epochs = [ 16, 22],
        total_epochs = 24,
    ),
)