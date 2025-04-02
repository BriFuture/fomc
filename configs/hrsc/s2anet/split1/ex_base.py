_base_ = ["ds_s2anet.py", "../rt_s2anet.py", "../model_s2anet_base.py"]

_split=1

experiment = dict(
    work_dir=f"checkpoints/hrsc/s2anet/split{_split}/base",
)

train = dict(
    solver = dict(
        learning_rate = 0.015,
        decay_epochs = [ 24, 33],
        decay_rate = 0.2,
        warmup_steps = 100,
        total_epochs = 36,
    ),
    eval_checkpoints=5,
)