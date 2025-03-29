import argparse

from retrieve_data_bet_boosted.retriever_bet_boosted_betclic import \
    RetrieverBetBoostedBetclic
from utils.tools import load_config


def main(config_path, env_path):
    """Retrieve the boosted odds thanks to the betboosted website and fill the database."""
    config = load_config(config_path, env_path)
    list_websites = {"betclic": RetrieverBetBoostedBetclic}
    for site, retriever_obj in list_websites.items():
        print(f"\nRetrieving data for {site} on Bet Boosted :")
        retriever_wepari = retriever_obj(**config["DB_VPS"], global_retrieve=config["SPECIFIC"]["global_retrieve"], table=site)
        retriever_wepari()


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

    # python3 retrieve_data_bet_boosted/main_retrieve_bet_boosted.py --config_path config/config_gagou.yaml --env_path config/.env.gagou