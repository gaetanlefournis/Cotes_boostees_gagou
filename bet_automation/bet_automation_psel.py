import asyncio
import time

import undetected_chromedriver as uc

from boosted_odds.bet_automation.bet_automation_winamax import \
    BetAutomationWinamax
from boosted_odds.boosted_odds_psel import BoostedOddsPSEL
from boosted_odds.boosted_odds_winamax import BoostedOddsWinamax
from boosted_odds.database.main_database import Database
from boosted_odds.telegram_bot.main_telegram import TelegramBot
from utils.human_behavior import HumanBehavior
from utils.tools import load_config


class BetAutomationPSEL():
    """Class to automate the betting on PSEL website. 

    The class will bet on
    the odds with a pending status in the database.

    Args:
        config_path (str): Path to the configuration file.
        env_path (str): Path to the environment file.
    """

    def __init__(
        self,
        config_path: str,
        env_path: str,
    ) -> None:
        self.config = None
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
        self.driver = uc.Chrome(
            headless=self.config["BO"]["headless"],
            user_agent=self.config["BO"]["user_agent"],
            use_subprocess=False,
        )
        self.human_behavior = HumanBehavior(self.driver)
        self.database = Database(**self.config["DB"])

    def main(self) -> None:
        """Main function to start the betting on PSEL. 

        The program will first retrieve the boosted odds for PSEL website. Then it will put them in the database and send the non already seen to telegram. Finally, it will bet on the odds with a pending status. 
        The program will run for each website in the config file.
        """
        # Retrieve the odds
        print(f"Starting time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        boosted_odds = BoostedOddsPSEL(**self.config["BO"]["parameters"], **self.config["DB"])
        final_bets_PSEL = asyncio.run(boosted_odds.main())

        # Bet on the odds
        if len(final_bets_PSEL) > 0:
            print(f"\nBet on {len(final_bets_PSEL)} odd(s) for PSEL :")
            boosted_odds = BetAutomationPSEL(**self.config["BO"]["parameters"], **self.config["DB"])
            final_list_bet_psel = asyncio.run(boosted_odds.main())

    def close(self) -> None:
        """Close the driver"""
        self.driver.quit()

    def run(self) -> None:
        """Main function to run the connection"""
        try:
            self.main()
        except:
            print("Unable to run the main function")
        self.close()