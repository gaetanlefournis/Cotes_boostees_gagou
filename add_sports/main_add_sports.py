import argparse

from add_sports.add_sports_manual import AddSportsManual
from add_sports.add_sports_telegram import AddSportsTelegram
from utils.tools import load_config


def main(config_path, env_path):
    config = load_config(config_path, env_path)
    if config["SPECIFIC"]["add_sport"] == "manual":
        add_sport = AddSportsManual(**config["DB"], table_name=config["SPECIFIC"]["table_name"])
        add_sport()
    elif config["SPECIFIC"]["add_sport"] == "telegram":
        add_sport = AddSportsTelegram(**config["DB"], **config["BO"]["parameters"], table_name=config["SPECIFIC"]["table_name"])
        add_sport()
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

    args = parser.parse_args()
    main(args.config_path, args.env_path)

    # python3 add_sports/main_add_sports.py --config_path config/config_gagou.yaml --env_path config/.env.gagou