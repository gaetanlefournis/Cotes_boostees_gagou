import time

import undetected_chromedriver as uc

from boosted_odds.boosted_odds_object.boosted_odds_object import \
    BoostedOddsObject
from boosted_odds.connection.connection_winamax import ConnectionWinamax
from utils.abstract import AbstractBettor
from utils.constants import URL_BOOSTED_ODDS_WINAMAX
from utils.human_behavior import HumanBehavior
from utils.tools import bet_on_Winamax, click_on_odd_button, close_right_panel


class BettorWinamax(AbstractBettor):
    """Class to bet on the boosted odds once we know they are good."""
    def __init__(
        self,
        driver : uc.Chrome,
        human_behavior : HumanBehavior,
        connection_username : str,
        connection_password : str,
        connection_day : str,
        connection_month : str,
        connection_year : str,
        amount_golden : float = 10,
        amount_silver : float = 2,
    ) -> None:
        self.driver = driver
        self.human_behavior = human_behavior
        self.connection_username = connection_username
        self.connection_password = connection_password
        self.connection_day = connection_day
        self.connection_month = connection_month
        self.connection_year = connection_year
        self.amount_golden = amount_golden
        self.amount_silver = amount_silver
        self.connection_object = None

    def _initiate(self) -> None:
        """Instantiate the Winamax Connection Object"""
        self.connector = ConnectionWinamax(self.driver, self.connection_username, self.connection_password, self.connection_day, self.connection_month, self.connection_year)

    def _connection_to_website(self) -> None:
        """Connect to the website with the username and password"""
        print("Trying to connect to Winamax")
        self.connector.run()
        # is_connected = self.connector._is_connected()
        # return is_connected
    
    def _bet_on_boosted_odd(self, boosted_odd : BoostedOddsObject) -> None:
        """Bet on the bet"""
        #assert right panel is closed
        closed = close_right_panel(self.driver)
        if not closed:
            raise Exception("Cannot close the right panel")
        self.human_behavior.gradual_scroll(boosted_odd.boosted_odd)
        clicked = click_on_odd_button(boosted_odd.boosted_odd)
        # Wait for the right column to load
        time.sleep(5)
        
        if clicked:
            # Bet on the boosted odd
            betted = bet_on_Winamax(self.driver, self.amount_golden if boosted_odd.golden == 'gold' else self.amount_silver)
            #close the right panel
            close_right_panel(self.driver)
            return betted
        return False

    def run(self, list_boosted_odds : list[BoostedOddsObject]) -> None:
        """Main function to bet on boosted odds on Winamax website"""
        try:
            self._initiate()
            self._connection_to_website()
            # Once connected, we go to the boosted odds url
            self.driver.get(URL_BOOSTED_ODDS_WINAMAX)
            for boosted_odd in list_boosted_odds:
                betted = self._bet_on_boosted_odd(boosted_odd)
                if not betted : 
                    print(f"couldn't bet on {boosted_odd.title} - {boosted_odd.sub_title}. Maybe the odd is not available anymore on {boosted_odd.website}")
        except Exception as e:
            print(f"There was a problem while betting on {boosted_odd.title} - {boosted_odd.sub_title} : {e}")