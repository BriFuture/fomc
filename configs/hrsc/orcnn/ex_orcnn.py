_base_ = ["ds_orcnn.py", "rt_orcnn.py", "../model_orcnn.py"]

experiment = dict(
    work_dir="checkpoints/hrsc/orcnn/all",
)

train = dict(
    solver = dict(
        learning_rate = 0.02
    )
)