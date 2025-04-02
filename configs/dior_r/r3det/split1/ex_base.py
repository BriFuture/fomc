_base_ = ["ds_r3det.py", "../rt_dior.py", "../model_r3det_base.py"]

_split=1

experiment = dict(
    work_dir=f"checkpoints/dior_r/r3det/split{_split}/base",
)

train = dict(
    solver = dict(
        learning_rate = 0.01,
        decay_epochs = [ 8, 11],
        total_epochs = 12,
    ),
)