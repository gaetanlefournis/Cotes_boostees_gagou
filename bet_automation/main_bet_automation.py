import argparse
import asyncio
import time

from bet_automation.bet_automation_winamax import BetAutomationWinamax
from boosted_odds.boosted_odds_psel import BoostedOddsPSEL
from boosted_odds.boosted_odds_winamax import BoostedOddsWinamax
from utils.tools import load_config


def main(config_path, env_path):
    config = load_config(config_path, env_path)

    # First retrieve the good odds and put them in the database
    print(f"Starting time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    for site in ["winamax", "PSEL"]:
        print(f"\nRetrieve odds for {site} :")
        if site == "winamax":
            boosted_odds = BoostedOddsWinamax(**config["BO"], **config["DB"], **config["TELEGRAM"])
            final_bets_winamax = asyncio.run(boosted_odds.main())
        elif site == "PSEL":
            boosted_odds = BoostedOddsPSEL(**config["BO"], **config["DB"], **config["TELEGRAM"])
            final_bets_PSEL = asyncio.run(boosted_odds.main())
        else :
            raise ValueError("The betting website is not recognized. Please check the config file.")
        
    # Then retreive the odds from the database and bet on the ones with a pending statut
    # if len(final_bets_winamax) > 0:
    #     print(f"\nBet on {len(final_bets_winamax)} odd(s) for winamax :")
    #     boosted_odds = BetAutomationWinamax(**config["BO"]["parameters"], **config["DB"], **config["CONNEXION"]["winamax"])
    #     final_list_bet_winamax = boosted_odds.main()
    # if len(final_bets_PSEL) > 0:
    #     print(f"\nBet on {len(final_bets_PSEL)} odd(s) for PSEL :")
    #     boosted_odds = BetAutomationPSEL(**config["BO"]["parameters"], **config["DB"], **config["CONNEXION"]["PSEL"])
    #     final_list_bet_psel = asyncio.run(boosted_odds.main())



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

    # python3 bet_automation/main_bet_automation.py --config_path config/config_gagou.yaml --env_path config/.env.gagou