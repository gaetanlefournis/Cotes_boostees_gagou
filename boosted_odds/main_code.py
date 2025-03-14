import argparse
import asyncio
import time

import undetected_chromedriver as uc
from bettor.bettor_psel import BettorPSEL
from bettor.bettor_winamax import BettorWinamax
from boosted_odds_object.boosted_odds_object import BoostedOddsObject
from database.main_database import Database
from retriever.retriever_psel import RetrieverPSEL
from retriever.retriever_winamax import RetrieverWinamax
from telegram_bot.main_telegram import TelegramBot

from utils.human_behavior import HumanBehavior
from utils.tools import load_config


class MainBoostedOdds():
    """Main class to process the boosted odds. 
    
    The program will first retrieve the boosted odds for each website. Then it will put them in the database and send the non already seen to telegram. Finally, it will bet on the odds with a pending statut. 
    The program will run for each website in the config file.
    
    Args:
        config_path (str): Path to the configuration file.
        env_path (str): Path to the environment file.
    """

    def __init__(
        self, 
        config_path : str, 
        env_path : str
    ) -> None:
        self.config = None
        self.class_creation_list = None
        self.driver = None
        self.human_behavior = None
        self.database = None
        self._initiate(config_path, env_path)
        
    def _initiate(self, config_path : str, env_path : str) -> None:
        """
        Initialize the main class. 
        Load the configuration and environment files and create the driver.
        """
        self.config = load_config(config_path, env_path)
        self.class_creation_list = {"winamax" : [RetrieverWinamax, BettorWinamax], "PSEL" : [RetrieverPSEL]}
        self.driver = uc.Chrome(headless=self.config["BO"]["headless"], user_agent=self.config["BO"]["user_agent"], use_subprocess=False)
        self.human_behavior = HumanBehavior(self.driver)
        self.database = Database(**self.config["DB"])
        self.telegram = TelegramBot(**self.config["TELEGRAM"])

    async def main_retrieve(self) -> list[BoostedOddsObject]:
        """Main function to start the boosted odds. 
        The program will first retrieve the boosted odds for each website. 
        """
        list_boosted_odds = []
        # First step : retrieve the boosted odds of every website
        for site in self.class_creation_list.keys():
            print(f"\nRetrieve odds for {site} :")
            retriever = self.class_creation_list[site][0](driver = self.driver)
            list_boosted_odds_partial = retriever.run()
            list_boosted_odds += list_boosted_odds_partial

        # Second step : Send a message to telegram if it's the first time seeing this odd
        # and put the boosted odds in the database
        for boosted_odd in list_boosted_odds:
            if not self.database.already_in_db(boosted_odd.dictionary):
                await self.telegram.send_boosted_odd_to_telegram(boosted_odd.dictionary)
                self.database.insert(boosted_odd.dictionary)

        return list_boosted_odds

    def main_bet(self, list_boosted_odds : list[BoostedOddsObject]) -> None:
        """Main function to start the betting. 

        The program will bet on the odds with a pending statut in the database. 
        The program will run for each website in the config file.
        """
        for site in self.class_creation_list.keys():
            if len(self.class_creation_list[site]) > 1:

                # Take the odds only for this website, only if not already bet on it
                new_list_boosted_odds = [obj for obj in list_boosted_odds if (obj.website == site and not self.database.already_bet_statut(obj.dictionary))]

                # Create the bettor only if there is a new bet to take.
                if len(new_list_boosted_odds) >= 1:
                    print(f"\nBet on odds for {site} :")
                    bettor = self.class_creation_list[site][1](driver = self.driver, human_behavior = self.human_behavior, **self.config["CONNECTION"][site])
                    bettor.run(new_list_boosted_odds)

            # Update in the database all the 'PENDING' bets statuts into 'BETTED'
            for boosted_odd in new_list_boosted_odds:
                self.database.update_bet_statut(boosted_odd.dictionary)

    def _close_all(self):
        """Close the driver"""
        self.driver.close()
        self.driver.quit()
        self.database.close()


    def run(self) -> None:
        """Main function to start the boosted odds and the betting. 

        The program will first retrieve the boosted odds for each website. Then it will put them in the database and send the non already seen to telegram. Finally, it will bet on the odds with a pending statut. 
        The program will run for each website in the config file.
        """
        print(f"Starting time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        
        try : 
            list_boosted_odds = asyncio.run(self.main_retrieve())
            self.main_bet(list_boosted_odds)
        except Exception as e:
            print(f"There was a problem in the boosted_odds program : {e}")
        finally:
            self._close_all()


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
    automate = MainBoostedOdds(args.config_path, args.env_path)
    automate.run()
    

    # python3 boosted_odds/main_code.py --config_path config/config_gagou.yaml --env_path config/.env.gagou 