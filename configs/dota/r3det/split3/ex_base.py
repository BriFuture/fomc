_base_ = ["ds_r3det.py", "../rt_dota.py", "../model_r3det_base.py"]

_split=3

experiment = dict(
    work_dir=f"checkpoints/dota/r3det/split{_split}/base",
)

train = dict(
    solver = dict(
        learning_rate = 0.01,
        decay_epochs = [ 12, 16],
        total_epochs = 18,
    ),
)