import datetime
import time
from decimal import Decimal
from typing import Any, List, Tuple

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text

from boosted_odds.boosted_odds_winamax import BoostedOddsWinamax
from boosted_odds.connection.connection_winamax import ConnectionWinamax
from utils import constants
from utils.abstract import AbstractBetAutomation
from utils.human_behavior import HumanBehavior
from utils.tools import bet_on_Winamax, click_on_odd_button, close_right_panel


class BetAutomationWinamax(AbstractBetAutomation):
    def __init__(
        self,
        db_database : str,
        db_user : str,
        db_password : str,
        db_host : str,
        db_port : str,
        db_table : str,
        headless : str,
        token_telegram : str,
        chat_id_telegram : str,
        connection_username : str,
        connection_password : str,
        connection_day : str,
        connection_month : str,
        connection_year : str,
        amount_golden : float,
        amount_silver : float,
        **kwargs : Any,
    ) -> None:
        self.db_database = db_database
        self.db_user = db_user
        self.db_password = db_password
        self.db_host = db_host
        self.db_port = db_port
        self.db_table = db_table
        self.connection_username = connection_username
        self.connection_password = connection_password
        self.connection_day = connection_day
        self.connection_month = connection_month
        self.connection_year = connection_year
        self.amount_golden = amount_golden
        self.amount_silver = amount_silver
        self.url_connection = constants.URL_CONNEXION_WINAMAX
        self.url_boosted_odds = constants.URL_BOOSTED_ODDS_WINAMAX
        self.WEBSITE = "winamax"
        self.headless = headless
        self.user_agent = "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US) AppleWebKit/532.0 (KHTML, like Gecko) Chrome/3.0.195.38 Safari/532.0"
        self.token_telegram = token_telegram
        self.chat_id_telegram = chat_id_telegram
        self.driver = None
        self.engine = None
        self.session = None
        self.to_bet_list = None
        self.boosted_odds_list = None
        self.is_connected = False
        self.Retriever_boosted_odds = None
        
    def _instantiate(self) -> None:
        """Instantiate the driver and the database engine"""
        self.driver = uc.Chrome(headless=self.headless, use_subprocess=False, user_agent=self.user_agent)
        self.engine = create_engine(
            f"mysql+mysqlconnector://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_database}"
        )
        self.session = sessionmaker(bind=self.engine)()
        self.connector = ConnectionWinamax(self.driver, self.connection_username, self.connection_password, self.connection_day, self.connection_month, self.connection_year)
        self.human_behavior = HumanBehavior(self.driver)

    def connection_to_website(self) -> None:
        """Connect to the website with the username and password"""
        print("Trying to connect to Winamax")
        if self.connector is None:
            print("No connection object initialized")
            return False

        self.connector.run()
        self.is_connected = self.connector._is_connected()
        return self.is_connected

    def retrieve_pending_bets(self) -> None:
        """For every bet with statut PENDING, we retrieve the bet"""
        query = text(f"SELECT * FROM {self.db_table} WHERE statut = 'PENDING'")
        result = self.session.execute(query)
        self.to_bet_list = []
        for row in result:
            bet = {'website': row[0], 'sport': row[1], 'title': row[2], 'sub_title': row[3], 'old_odd': row[4], 'odd': row[5], 'golden': row[6]}
            self.to_bet_list.append(bet)

    def _get_sports_classes(self):
        try:
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located(
                    (By.XPATH, "//div[@data-testid='leftColumn']")
                )
            )
            left = self.driver.find_element(
                By.XPATH, "//div[@data-testid='leftColumn']"
            ).find_element(By.XPATH, ".//div[2]//div[3]//div[2]")

            # Get the list of the sports
            sports = left.find_elements(By.XPATH, "./*")
            _sport_dict = {}
            for sport in sports:
                try:
                    sport_text = sport.find_element(By.XPATH, ".//a//div[2]//span").get_attribute("innerText")
                    _sport_dict[sport_text] = (
                        sport.find_element(By.XPATH, ".//a//div[1]//div//div//div")
                        .get_attribute("class")
                        .split(" ")
                    )
                except:
                    pass


            # For each sport key keep only elements in the list that are unique amongst all the lists of the dict
            new_dict = {}
            for key, value in _sport_dict.items():
                for key2, value2 in _sport_dict.items():
                    if key != key2:
                        value = [x for x in value if x not in value2]
                        if len(value) == 1:
                            break
                
                if not value:
                    new_dict["other"] = key
                else:
                    new_dict[value[0]] = key
            self._sport_dict = new_dict
                
        except:
            print("Error while getting the sports classes")

    def _get_infos_from_boosted_odd(
        self,
        boosted_odd: WebElement,
    ):
        """Get the infos from a boosted odd

        Args:
            boosted_odd (WebElement): The boosted odd
        Returns:
            dict[str, str, str, str, str, bool]: infos of the boosted odd
        """
        # Split the text
        text = boosted_odd.get_attribute("innerText").split("\n")
        heure, title, sub_title, old_odd, odd = (
            text[0],
            text[1],
            text[2],
            text[3],
            text[4],
        )

        # if data-testid = boosted-odds-countdown, then it's a countdown
        is_countdown = len(boosted_odd.find_elements(By.XPATH, "//*[@data-testid='boosted-odds-countdown']")) > 0

        # Find the sport with the logo
        sport = boosted_odd.find_element(
            By.XPATH, ".//div//div//div//div[1]//div[1]//div//div"
        ).get_attribute("class")
        
        
        classes_sport = sport.split(" ")
        for class_sport in classes_sport:
            if class_sport in self._sport_dict:
                sport = self._sport_dict[class_sport]
                break

        # Get the date
        jour = heure.split()[0]
        heure = heure.split()[-1]
        if "Demain" in jour:
            date = datetime.datetime.now() + datetime.timedelta(days=1)
        elif "Hier" in jour:
            date = datetime.datetime.now() - datetime.timedelta(days=1)
        else:
            date = datetime.datetime.now()

        if is_countdown:
            minutes, seconds = map(int, heure.split(":"))
            match_time = datetime.datetime.now() + datetime.timedelta(minutes=minutes, seconds=seconds)

            # Handle edge case where countdown crosses to the next day
            if match_time.day > date.day:
                date += datetime.timedelta(days=1)

            date = date.replace(hour=match_time.hour, minute=match_time.minute, second=match_time.second)
        else:
            heure = datetime.datetime.strptime(heure, "%H:%M")
            date = date.replace(hour=heure.hour, minute=heure.minute)

        # Process the odds
        old_odd = Decimal(old_odd.replace(",", ".")) if old_odd != "" or None else None
        odd = Decimal(odd.replace(",", ".")) if odd != "" or None else None

        # Check if the odd is gold
        proba_diff = 1 / old_odd - 1 / odd
        if proba_diff > 0.11:
            golden = "gold"
        else:
            golden = "silver"

        # Create the bet
        bet = {
            "website": "winamax",
            "title" : title,
            "sub_title" : sub_title,
            "old_odd" : old_odd,
            "odd" : odd,
            "golden" : golden,
            "sport" : sport,
            "date" : date,
        }
        return bet

    def retrieve_boosted_odds(self):
        """Retrieve the boosted odds from the Winamax website

        Returns:
            Tuple[List[WebElement],List[Dict[str,str]]]: list of boosted odds, list of tuples (sport, odd)
        """
        
        # Get sport classes
        self._get_sports_classes()
        # Find the boosted odds on the page
        try:
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_all_elements_located(
                    (
                        By.XPATH,
                        "//div[@class='ReactVirtualized__Grid__innerScrollContainer']/*",
                    )
                )
            )
            time.sleep(1)
            boosted_odds = self.driver.find_elements(
                By.XPATH,
                "//div[@class='ReactVirtualized__Grid__innerScrollContainer']/*",
            )
            
        except Exception as _:
            print("Error while retrieving the boosted odds")
            boosted_odds = []

        list_bet = []
        for boosted_odd in boosted_odds:
            # Get the infos of the boosted odd
            bet = self._get_infos_from_boosted_odd(boosted_odd)
            list_bet.append(bet)
                
        return boosted_odds, list_bet

    def find_bet_on_winamax(self, bet) -> WebElement:
        """Find the bet on the website"""
        def compare_infos_with_db(infos, bet):
            return all(infos[key] == bet[key] for key in ['website', 'sport', 'title', 'sub_title', 'old_odd', 'odd'])
        
        for i, infos in enumerate(self.info_from_boosted_odds_list):
            if compare_infos_with_db(infos, bet):
                return self.boosted_odds_list[i]
        return None

    def bet_on_bet(self, boosted_odd, bet) -> None:
        """Bet on the bet"""
        #assert right panel is closed
        closed = close_right_panel(self.driver)
        if not closed:
            raise Exception("Cannot close the right panel")
        self.human_behavior.gradual_scroll(boosted_odd)
        clicked = click_on_odd_button(boosted_odd)
        # Wait for the right column to load
        time.sleep(5)
        
        if clicked:
            # Bet on the boosted odd
            betted = bet_on_Winamax(self.driver, self.amount_golden if bet['golden'] == 'gold' else self.amount_silver)
            #close the right panel
            close_right_panel(self.driver)
            return betted
        return False

    def change_bet_statut(self, bet) -> None:
        """Depending on the answer of bet_on_bet, we change the statut"""
        query = text(f"UPDATE {self.db_table} SET statut = 'BETTED' WHERE website = :website AND sport = :sport AND title = :title AND sub_title = :sub_title AND old_odd = :old_odd AND odd = :odd")
        self.session.execute(query, {'website': bet['website'], 'sport': bet['sport'], 'title': bet['title'], 'sub_title': bet['sub_title'], 'old_odd': bet['old_odd'], 'odd': bet['odd']})
        self.session.commit()

    def main(self) -> None:
        """Main function to run the automation"""
        try:
            self._instantiate()
            self.connection_to_website()
            self.retrieve_pending_bets()
            self.find_bets_on_winamax()
            for bet in self.to_bet_list:
                boosted_odd = self.find_bet_on_winamax(bet)
                if boosted_odd is not None:
                    is_bet = self.bet_on_bet(boosted_odd, bet)
                    if is_bet:
                        self.change_bet_statut(bet)
        except Exception as e:
            print(f"Error in the main function: {e}")
        finally:
            self.driver.quit()
            self.session.close()

    def close(self):
        """Close properly the driver and the database session"""
        self.driver.quit()
        self.session.close()
