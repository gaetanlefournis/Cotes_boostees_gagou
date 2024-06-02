import argparse
import asyncio
import random
import time

from boosted_odds_psel import BoostedOddsPSEL
from boosted_odds_winamax import BoostedOddsWinamax

from utils.tools import load_config


def main(config_path, env_path):
    config = load_config(config_path, env_path)
    bet_history = []
    while True:
        for site in config["BO"]["websites"]:
            print(f"\nBoosted odds for {site} :")
            if site == "WINAMAX":
                boosted_odds = BoostedOddsWinamax(bet_history=bet_history, **config["BO"]["parameters"])
                bet_history = asyncio.run(boosted_odds.main())
            elif site == "PSEL":
                boosted_odds = BoostedOddsPSEL(bet_history=bet_history, **config["BO"]["parameters"])
                bet_history = asyncio.run(boosted_odds.main())
            else :
                raise ValueError("The betting website is not recognized. Please check the config file.")
        wait = random.randint(12 * 60, 15 * 60)
        time.sleep(wait)


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

    # python3 boosted_odds/main_boosted_odds.py --config_path config/config_gagou.yaml --env_path config/.env.gagou