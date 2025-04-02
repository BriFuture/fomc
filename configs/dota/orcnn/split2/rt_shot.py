_base_ = "../rt_dota.py"

experiment = dict(
    type="ExpFewshotTrain",
    load_from = "checkpoints/weights/dota/orcnn/split2/softmax/model_reset_surgery.pth",
)
train = dict(
    log_interval = 10,
    solver = dict(
        warmup_steps = 20,
        learning_rate = 0.001,
    ),
    data_loader = dict(
        samples_per_gpu = 8,
        workers_per_gpu = 4,
    ),
    save_checkpoints = 1000,
)