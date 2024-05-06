import argparse

from retriever_psel import RetrieverPSEL
from retriever_winamax import RetrieverWinamax

from utils.tools import load_config


def main(config_path, env_path):
    config = load_config(config_path, env_path)
    if config["betting_website"] == "WINAMAX":
        retriever_wepari = RetrieverWinamax(**config)
        retriever_wepari()
    elif config["betting_website"] == "PSEL":
        retriever_wepari = RetrieverPSEL(**config)
        retriever_wepari()
    else :
        raise ValueError("The betting website is not recognized. Please check the config file.")


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

    # python3 retrieve_data_wepari/main_retrieve.py --config_path config/config_gagou.yaml --env_path config/.env.gagou