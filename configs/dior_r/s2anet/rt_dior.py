_base_ = "../rt_dior.py"

train = dict(
    solver = dict(
        decay_epochs = [290,],
        total_epochs = 300,
        warmup_steps = 10,
        learning_rate = 0.01,
    ),
)