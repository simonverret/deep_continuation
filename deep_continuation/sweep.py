from deep_continuation.training import train_mlp

import wandb
wandb.login()


# 2: Define the search space
sweep_configuration = {
    'method': 'random',
    'metric': 
    {
        'goal': 'minimize', 
        'name': 'score'
    },
    'parameters': 
    {
        'x': {'max': 0.1, 'min': 0.01},
        'y': {'values': [1, 3, 7]},
        # dataset options
        'noise': 0.0, 
        'standardize': True, 
        # datafile options
        'name': "unbiased",
        'path': None,
        'num_std': 1,
        'num_beta': 1,
        'Nwn': 128,
        'beta': 30,
        'Nw': 512,
        'wmax': 20,
        'fixstd': False,
        # dataloader options
        'lr ':  8e-5,
        'batch_size ':  523,
        # model options
        'layers': [128, 952, 1343, 1673, 1722, 512],
        # scheduler options
        'factor': 0.216, 
        'patience': 5, 
        'min_lr': 1e-10,
        # training loop options
        'n_epochs ':  800,
        'early_stop_limit ':  40,
        'warmup ':  True,
        'use_wandb ':  False,
    }
}

# 3: Start the sweep
sweep_id = wandb.sweep(
    sweep=sweep_configuration, 
    entity="simonverret", 
    project="taac",
)

wandb.agent(sweep_id, function=train_mlp, count=30)