experiment = dict(
    epochs = 10,
    progress_train = True,
    seed = 1000,
    work_dir = None,
    load_from = "",
    type = "ExpTrain",
)

train = dict(
    save_checkpoints = 50,
    eval_checkpoints = 5,
    eval_start = 0,
    auto_copy_best = True,
    fp16 = dict(
        enabled = False,
        skip_modules = None,
        loss_scale=512.,
    ),
    # batch_size = 4, ###  use samples_per_gpu instead
    # num_workers = 2,
    data_loader = dict(
        shuffle = True,
        pin_memory = True,
        sampling_strategy = 'instance_balanced',
        samples_per_gpu = 4,
        workers_per_gpu = 4,
    ),
    solver = dict(
        optimizer = 'SGD',
        learning_rate = 0.01,
        momentum = 0.9,
        decay_factor = 0.1,
        weight_decay = 0.0001,
        weight_decay_norm = 0.0,
        decay_epochs = [ 30, ],
        total_epochs = 40,
        decay_rate = 0.1,
        warmup_steps = 100,

        nesterov = False, # only for SGD.
        lr_scheduler = "cosine", # available schedulers are list in 'scheduler_args' below. please remember to update scheduler_args when number of epoch changes.
        adam_betas = (0.9, 0.999),
        adamw_betas = [0.001, 0.999],
    ),
)

val = dict(

    fp16 = dict(
        enabled = False,
        skip_modules = None,
        loss_scale=512.,
    ),
    # batch_size = 4,
    # num_workers = 2,
    data_loader = dict(
        shuffle = False,
        pin_memory = True,
        sampling_strategy = 'instance_balanced',
        samples_per_gpu = 4,
        workers_per_gpu = 4,
    )
)

test = dict(
    batch_size = 2,
    shuffle = False,
    fp16 = dict(
        enabled = False,
        skip_modules = None,
        loss_scale=512.,
    ),
)


dist = dict(
    distributed = False,  ### default is False, do not change it mannually, changed by program
    rank = 0,  ### default is False, do not change it
    world_size = 1,  ### single gpu, default is 1, do not change it
    ### can be edited
    gpu_nums = 0,
    gpu_ids = [0],
    backend = 'nccl',
    addr = '127.0.0.1', # address used to set up distributed training, torchrun 通过命令行指定
    port = 29100, # port used to set up distributed training
    launcher = 'pytorch',
    find_unused_parameters = True,
)

