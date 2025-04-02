_base_ = "../rt_base.py"
experiment = dict(
    type="ExpMMDetTrain",
    modules = {
        "base": [
            "gdet.engine.exp_few_shot", 
            "mmrotate" ,
        ]
    },
    work_dir="checkpoints/hrsc/s2anet/all",
)

train = dict(
    save_checkpoints = 36,
    eval_checkpoints=1,
    log_interval = 50,
    data_loader = dict(
        samples_per_gpu = 8,
        workers_per_gpu = 4,

    ),
    fp16 = dict(
        enabled = False,
        start_iter = 100,
    ),
    solver = dict(
        learning_rate = 0.01,
        warmup_steps = 500,
        momentum = 0.9,
        weight_decay=0.0001,
        optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2)),
        decay_epochs = [ 8, 11],
        total_epochs = 12,
        decay_rate = 0.1,
        nesterov = False, # only for SGD.
        lr_scheduler = "cosine", # available schedulers are list in 'scheduler_args' below. please remember to update scheduler_args when number of epoch changes.
    ),
)

val = dict(
    fp16 = dict(
        enabled = False,
    ),
    data_loader = dict(
        samples_per_gpu = 8,
        workers_per_gpu = 4,
    )
)


