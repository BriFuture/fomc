_base_ = ["ds_s2anet.py", "../rt_dota.py", "../model_s2anet_base.py"]

_split=1

experiment = dict(
    work_dir=f"checkpoints/dota/s2anet/split{_split}/base",
)

train = dict(
    solver = dict(
        learning_rate = 0.008,
        decay_epochs = [ 8, 11],
        total_epochs = 12,
        warmup_steps = 200,
    ),
)