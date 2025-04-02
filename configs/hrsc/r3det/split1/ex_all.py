_base_ = ["ds_r3det_shot.py", "rt_shot.py", "../model_r3det_shot.py"]

experiment = dict(
    work_dir="checkpoints/hrsc/r3det/split1/all",
)

train = dict(
    solver = dict(
        learning_rate = 0.005,
        decay_epochs = [ 24, 33],
        decay_rate = 0.1,
        warmup_steps = 100,
        total_epochs = 36,
    ),
    log_interval = 5,
)