_base_ = ["ds_orcnn_shot.py", "../rt_orcnn.py", "../model_orcnn_all.py"]

experiment = dict(
    work_dir="checkpoints/hrsc/orcnn/split1/all",
)

train = dict(
    solver = dict(
        learning_rate = 0.02,
        decay_epochs = [ 24, 33],
        decay_rate = 0.1,
        warmup_steps = 100,
        total_epochs = 36,
    ),
    log_interval = 5,
)