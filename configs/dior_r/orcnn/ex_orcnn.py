_base_ = ["ds_orcnn.py", "rt_dior.py", "../model_orcnn.py"]

experiment = dict(
    work_dir="checkpoints/dior_r/orcnn/all",
)

train = dict(
    solver = dict(
        learning_rate = 0.02
    )
)