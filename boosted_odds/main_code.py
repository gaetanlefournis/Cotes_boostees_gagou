"""MAIN CODE"""

import argparse
import asyncio
import random
import time

import undetected_chromedriver as uc

from boosted_odds.bettor.bettor_psel import BettorPSEL
from boosted_odds.bettor.bettor_winamax import BettorWinamax
from boosted_odds.boosted_odds_object.boosted_odds_object import \
    BoostedOddsObject
from boosted_odds.database.main_database import Database
from boosted_odds.retriever.retriever_betclic import RetrieverBetclic
from boosted_odds.retriever.retriever_psel import RetrieverPSEL
from boosted_odds.retriever.retriever_unibet import RetrieverUnibet
from boosted_odds.retriever.retriever_winamax import RetrieverWinamax
from boosted_odds.telegram_bot.main_telegram import TelegramBot
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

    def _initiate(self, config_path: str, env_path: str) -> None:
        """
        Initialize the main class. 
        Load the configuration and environment files and create the driver.
        """
        self.config = load_config(config_path, env_path)
        self.class_creation_list = {"winamax" : [RetrieverWinamax, BettorWinamax], "PSEL" : [RetrieverPSEL], "betclic": [RetrieverBetclic], "unibet" : [RetrieverUnibet]}

        # Driver setup
        options = uc.ChromeOptions()

        if not self.config['BO']['headless']:  # Ensure headless is False
            print("Debug Mode: Headless Disabled")
            options.add_argument("--disable-blink-features=AutomationControlled")  # Prevent detection
            options.add_argument("--disable-gpu")
            options.add_argument("--force-device-scale-factor=1")
        else:
            options.add_argument("--headless=new")  # Use "new" for better compatibility

        selected_user_agent = random.choice(self.config['BO']['user_agents'])
        options.add_argument(f"user-agent={selected_user_agent}")
        options.add_argument("--use_subprocess=True")
        options.add_argument("--no-sandbox")
        options.add_argument("--start-maximized")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--remote-debugging-port=9222")

        # Anti-detection improvements
        options.add_argument("--disable-blink-features=AutomationControlled")  # Prevent bot detection
        options.add_argument("--disable-infobars")  # Disable "Chrome is being controlled by automated software"
        options.add_argument("--disable-extensions")  # Disable extensions
        options.add_argument("--disable-software-rasterizer")  # Disabling the software renderer
        options.add_argument("--disable-features=VizDisplayCompositor")  # Prevents UI blocking

        # Initialize Chrome
        self.driver = uc.Chrome(options=options)

        # The rest
        self.human_behavior = HumanBehavior(self.driver)
        self.database = Database(**self.config["DB_VPS"])
        self.telegram = TelegramBot(**self.config["TELEGRAM"])

    async def main_retrieve(self) -> list[BoostedOddsObject]:
        """Main function to start the boosted odds. 
        The program will first retrieve the boosted odds for each website. 
        """
        final_good_boosted_odds = []
        final_all_boosted_odds = []
        # First step : retrieve the boosted odds of every website
        for site in self.class_creation_list:
            print(f"\nRetrieve odds for {site} :")
            retriever = self.class_creation_list[site][0](driver = self.driver)
            all_boosted_odds, good_boosted_odds = retriever.run()
            final_good_boosted_odds += good_boosted_odds
            final_all_boosted_odds += all_boosted_odds
        
        print(f"length all boosted odds : {len(final_all_boosted_odds)}")

        # Second step : Insert the boosted odds in the database corresponding to each odd if not already in
        for boosted_odd in final_all_boosted_odds:
            if not self.database.already_in_db(boosted_odd.dictionary, table=boosted_odd.website):
                self.database.insert(boosted_odd.dictionary, table=boosted_odd.website)

        # Third step : Send a message to telegram if it's the first time seeing this odd
        # and put the boosted odds in the database
        for boosted_odd in final_good_boosted_odds:
            if not self.database.already_in_db(boosted_odd.dictionary, table=self.config["DB_VPS"]["db_table"]):
                await self.telegram.send_boosted_odd_to_telegram(boosted_odd.dictionary)
                self.database.insert(boosted_odd.dictionary, table=self.config["DB_VPS"]["db_table"])

        return final_good_boosted_odds

    def main_bet(self, list_boosted_odds : list[BoostedOddsObject]) -> None:
        """Main function to start the betting. 

        The program will bet on the odds with a pending statut in the database. 
        The program will run for each website in the config file.
        """
        for site in self.class_creation_list:
            if len(self.class_creation_list[site]) > 1:

                # Take the odds only for this website, only if not already bet on it
                new_list_boosted_odds = [obj for obj in list_boosted_odds if (obj.website == site and not self.database.already_bet_statut(obj.dictionary, table=self.config["DB_VPS"]["db_table"]))]

                # Create the bettor only if there is a new bet to take.
                if len(new_list_boosted_odds) >= 1:
                    print(f"\nBet on odds for {site} :")
                    bettor = self.class_creation_list[site][1](driver = self.driver, human_behavior = self.human_behavior, **self.config["CONNECTION"][site])
                    bettor.run(new_list_boosted_odds)

            # Update in the database all the 'PENDING' bets statuts into 'BETTED'
            for boosted_odd in new_list_boosted_odds:
                self.database.update_bet_statut(boosted_odd.dictionary, self.config["DB_VPS"]["db_table"])

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