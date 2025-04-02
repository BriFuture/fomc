_base_ = ["ds_s2anet_shot.py", "../rt_s2anet.py", "../model_s2anet_all.py"]

experiment = dict(
    work_dir="checkpoints/hrsc/s2anet/split1/all",
)

train = dict(
    solver = dict(
        learning_rate = 0.01,
        decay_epochs = [ 12, 16],
        total_epochs = 18,
    ),
    log_interval = 5,
)