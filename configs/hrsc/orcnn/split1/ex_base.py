_base_ = ["ds_orcnn.py", "../rt_orcnn.py", "../model_orcnn_base.py"]

_split=1

experiment = dict(
    work_dir=f"checkpoints/hrsc/orcnn/split{_split}/base",
)

train = dict(
    solver = dict(
        learning_rate = 0.012,
        decay_epochs = [ 24, 33],
        total_epochs = 36,
    ),
)