_base_ = ["ds_orcnn.py", "rt_dota.py", "../model_orcnn.py"]

experiment = dict(
    work_dir="checkpoints/dota/orcnn/all",
)

train = dict(
    solver = dict(
        learning_rate = 0.02
    )
)