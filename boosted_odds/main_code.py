import argparse
import asyncio
import time

import undetected_chromedriver as uc

from boosted_odds.database.main_database import Database
from boosted_odds.retriever.retriever_winamax import RetrieverWinamax
from boosted_odds.telegram_bot.main_telegram import TelegramBot
from utils.human_behavior import HumanBehavior
from utils.tools import load_config


class MainBoostedOdds():
    """Main class to process the boosted odds. 
    
    The program will first retrieve the boosted odds for each website. Then it will put them in the database and send the non already seen to telegram. Finally, it will bet on the odds with a pending status. 
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
        self.class_creation_list = {"winamax" : [RetrieverWinamax]}
        self.driver = uc.Chrome(headless=self.config["BO"]["headless"], user_agent=self.config["BO"]["user_agent"], use_subprocess=False)
        self.human_behavior = HumanBehavior(self.driver)
        self.database = Database(**self.config["DB"])
        self.telegram = TelegramBot(**self.config["TELEGRAM"])

    def main_retrieve(self) -> None:
        """Main function to start the boosted odds. 

        The program will first retrieve the boosted odds for each website. Then it will put them in the database and send the non already seen to telegram. Finally, it will bet on the odds with a pending status. 
        The program will run for each website in the config file.

        Args:
            config_path (str): Path to the configuration file.
            env_path (str): Path to the environment file.
        """
        # First step : retrieve the boosted odds of every website
        for site in self.class_creation_list.keys():
            print(f"\nRetrieve odds for {site} :")
            retriever = self.class_creation_list[site][0](driver = self.driver, human_behavior = self.human_behavior)
            list_boosted_odds = retriever.run()

            # Second step : Send a message to telegram if it's the first time seeing this odd
            # and put the boosted odds in the database
            for boosted_odd in list_boosted_odds:
                if not self.database.already_in_db(boosted_odd.dictionary):
                    self.telegram.send_boosted_odd_to_telegram(boosted_odd.dictionary)
                    self.database.insert(boosted_odd.dictionary)

    def main_bet(self) -> None:
        """Main function to start the betting. 

        The program will bet on the odds with a pending status in the database. 
        The program will run for each website in the config file.
        """
        for site in self.class_creation_list.keys():
            print(f"\nBet on odds for {site} :")
            better = self.class_creation_list[site][1](driver = self.driver, human_behavior = self.human_behavior, database = self.database, **self.config["CONNEXION"][site])
            better.main()

    def main(self) -> None:
        """Main function to start the boosted odds and the betting. 

        The program will first retrieve the boosted odds for each website. Then it will put them in the database and send the non already seen to telegram. Finally, it will bet on the odds with a pending status. 
        The program will run for each website in the config file.
        """
        print(f"Starting time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        
        self.main_retrieve()
        self.main_bet()





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
    automate.main_retrieve()
    

    # python3 boosted_odds/main_code.py --config_path config/config_gagou.yaml --env_path config/.env.gagou 