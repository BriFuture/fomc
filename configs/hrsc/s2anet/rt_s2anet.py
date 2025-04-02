_base_ = "../rt_hrsc.py"

train = dict(
    save_checkpoints = 100,
    eval_checkpoints=1,
    log_interval = 50,
    solver = dict(
        learning_rate = 0.01,
        decay_epochs = [ 24, 33],
        total_epochs = 36,
    ),
    data_loader = dict(
        samples_per_gpu = 8,
        workers_per_gpu = 4,
    ),
)