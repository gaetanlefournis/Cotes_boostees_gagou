import os

import dotenv
import yaml


def substitute_placeholders(value, env: dict):
    if isinstance(value, str) and value.startswith("$"):
        if env is None:
            return os.environ.get(value[2:-1])  # Fetch value from environment variable
        return env.get(value[2:-1])  # Fetch value from environment variable
    if isinstance(value, dict):
        return {k: substitute_placeholders(v, env) for k, v in value.items()}
    if isinstance(value, list):
        return [substitute_placeholders(v, env) for v in value]
    return value

def load_config(yaml_path: str, env_var_path: str = None):
    """Convert a yaml file to a dict

    Args:
        yaml (str): The path to the yaml file

    Returns:
        dict: The dict corresponding to the yaml file
    """
    env = None
    if env_var_path is not None:
        env = dotenv.dotenv_values(env_var_path)
    with open(yaml_path) as file:
        config = yaml.safe_load(file)

    config = substitute_placeholders(config, env)

    return config

def save_fig(fig, path: str):
    """Save a figure to a file, if the path doesn't exist, create it"""
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    fig.savefig(path)

    