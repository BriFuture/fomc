_base_ = ["ds_r3det.py", "../rt_r3det.py", "../model_r3det_base.py"]

_split=1

experiment = dict(
    work_dir=f"checkpoints/hrsc/r3det/split{_split}/base",
)

train = dict(
    solver = dict(
        learning_rate = 0.01,
        decay_epochs = [ 24, 33],
        warmup_steps = 100,
        total_epochs = 36,
    ),
    eval_checkpoints=5,
    log_interval = 10,
)