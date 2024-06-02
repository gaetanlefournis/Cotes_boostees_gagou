import argparse

from add_sports.add_sports_automatic import AddSportsAutomatic
from add_sports.add_sports_manual import AddSportsManual
from utils.tools import load_config


def main(config_path, env_path, word : str = "", sport : str = ""):
    config = load_config(config_path, env_path)
    if config["DB"]["add_sport"] == "manual" or (word == "" and sport == ""):
        add_sport = AddSportsManual(**config["DB"])
        add_sport()
    elif config["DB"]["add_sport"] == "automatic":
        add_sport = AddSportsAutomatic(**config["DB"])
        add_sport(word, sport)
    else :
        raise ValueError("The way of adding sports is not recognized. Please check the config file.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script with config and env file paths."
    )
    parser.add_argument(
        "--config_path",
        default="config.yaml",
        help="Path to the configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--env_path",
        default=".env.example",
        help="Path to the environment file (default: .env.example)",
    )

    parser.add_argument(
        "--word",
        default="",
        help="Word to add",
    )

    parser.add_argument(
        "--sport",
        default="",
        help="Sport to add",
    )

    args = parser.parse_args()
    main(args.config_path, args.env_path, args.word, args.sport)

    # python3 add_sports/main_add_sports.py --config_path config/config_gagou.yaml --env_path config/.env.gagou --sport "Football" --word "football"