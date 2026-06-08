import wandb
import os

def initialize_wandb_run(project_name, run_name, config):
    wandb.login(key=os.getenv("API_KEY"))

    if not isinstance(config, dict):
        config = vars(config)
    wandb.init(
        name=run_name,
        project=project_name,
        config=config,
    )
    return wandb
