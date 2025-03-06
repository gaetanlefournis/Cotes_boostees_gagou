import argparse
import asyncio
import time

from boosted_odds_psel import BoostedOddsPSEL
from boosted_odds_winamax import BoostedOddsWinamax

from utils.tools import load_config


def main(config_path, env_path) -> None:
    config = load_config(config_path, env_path)
    print(f"Starting time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    for site in config["BO"]["websites"]:
        print(f"\nBoosted odds for {site} :")
        if site == "winamax":
            boosted_odds = BoostedOddsWinamax(**config["BO"]["parameters"], **config["DB"])
            final_list_bet_winamax = asyncio.run(boosted_odds.main())
            print(final_list_bet_winamax)
        elif site == "PSEL":
            boosted_odds = BoostedOddsPSEL(**config["BO"]["parameters"], **config["DB"])
            final_list_bet_psel = asyncio.run(boosted_odds.main())
            print(final_list_bet_psel)
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

    # python3 boosted_odds/main_boosted_odds.py --config_path config/config_gagou.yaml --env_path config/.env.gagou