_base_ = ["ds_orcnn.py", "../rt_dota.py", "../model_orcnn_base.py"]

_split=2

experiment = dict(
    work_dir=f"checkpoints/dota/orcnn/split{_split}/base",
)

train = dict(
    solver = dict(
        learning_rate = 0.01,
        decay_epochs = [ 8, 11],
        total_epochs = 12,
    ),
)