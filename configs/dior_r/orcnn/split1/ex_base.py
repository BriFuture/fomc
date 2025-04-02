_base_ = ["ds_orcnn.py", "../rt_dior.py", "../model_orcnn_base.py"]

_split=1

experiment = dict(
    work_dir=f"checkpoints/dior_r/orcnn/split{_split}/base",
)

train = dict(
    solver = dict(
        learning_rate = 0.02,
        decay_epochs = [ 8, 11],
        total_epochs = 12,
    ),
)