import datetime
import re
import time
from decimal import Decimal

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from boosted_odds.boosted_odds_object.boosted_odds_object import \
    BoostedOddsObject
from utils.abstract import AbstractRetriever
from utils.constants import CONDITIONS_ON_SPORTS, URL_BOOSTED_ODDS_BETCLIC
from utils.tools import find_button_by_text


class RetrieverBetclic(AbstractRetriever):
    """Class to retrieve the boosted odds from Betclic. After the connection, the final goal is to return a list of boosted odds that respect the conditions in the constants file."""
    def __init__(
        self,
        driver : uc.Chrome = None,
        **kwargs,
    ) -> None:
        self.WEBSITE = "betclic"
        self.url_connexion = URL_BOOSTED_ODDS_BETCLIC
        self.conditions_on_sports = CONDITIONS_ON_SPORTS[self.WEBSITE]
        self._sport_dict = None
        self.final_list_bet = None
        self.list_boosted_odds_objects = []
        self.all_boosted_odds = None
        self.driver = driver

    def _initiate(self) -> None:
        """Create the driver if it doesn't exist yet"""
        pass

    def _load_page(self) -> None:
        """Load the page betclic.fr and close the popups and accept the cookies"""
        self.driver.get(self.url_connexion)
        try : 
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            time.sleep(5)
            self._close_first_popup()
            self._accept_cookies()
        except Exception as e:
            print("Error while loading page or while accepting the cookies : {}".format(e))

    def _close_first_popup(self) -> None:
        """Close the first popup"""
        close_button = find_button_by_text(self.driver, "close")
        if close_button:
            close_button.click()
            print("First popup closed")
        else:
            print("the button to close the first popup was not found")

    def _accept_cookies(self) -> None:
        """Accept the cookies or continue without accepting"""
        accept_button = find_button_by_text(self.driver, "Continuer sans accepter")
        if accept_button:
            accept_button.click()
            print("Cookies accepted")
        else:
            print("the button to accept the cookies was not found")

    def _get_sports_classes(self) -> None:
        """Get a dictionary with the sports and their classes. This will be useful to know afterwards the sports of the boosted odds.
        """
        try:
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located(
                    (By.XPATH, "//div[@class='list is-sportList']")
                )
            )
            left = self.driver.find_element(
                By.XPATH, "//div[@class='list is-sportList']"
            ).find_element(By.XPATH, ".//div[2]")

            # Get the list of the sports
            sports = left.find_elements(By.XPATH, "./*")
            _sport_dict = {}
            for sport in sports:
                try:
                    sport_text = sport.find_element(By.XPATH, ".//div//div[2]//div").get_attribute("innerText")
                    _sport_dict[sport_text] = (
                        sport.find_element(By.XPATH, ".//div//div[1]//span")
                        .get_attribute("class")
                        .split(" ")
                    )
                except:
                    pass
            # For each sport key keep only elements in the list that are unique amongst all the lists of the dict
            new_dict = {}
            for key, value in _sport_dict.items():
                new_dict[value[2]] = key
            self._sport_dict = new_dict
                
        except Exception as e:
            print("Error while getting the sports classes")
            print(e)

    def _get_infos_from_boosted_odd(
        self,
        boosted_odd: WebElement,
        title: str,
        sport: str,
        date: str,
    ):
        """Get the infos from a boosted odd

        Args:
            boosted_odd (WebElement): The boosted odd
        Returns:
            dict[str, str, str, str, str, bool]: infos of the boosted odd
        """
        text = boosted_odd.text.split("\n")
        print(text)
        if len(text) == 5:
            sub_title = text[1]
            old_odd = text[3]
            odd = text[4]
        else:
            raise Exception
        
        title = title
        sport = sport
        jour = date.split()[0]
        heure = date.split()[-1]
        if "Demain" in jour:
            date = datetime.datetime.now() + datetime.timedelta(days=1)
        elif "Hier" in jour:
            date = datetime.datetime.now() - datetime.timedelta(days=1)
        else:
            date = datetime.datetime.now()
        heure = datetime.datetime.strptime(heure.split()[-1], "%H:%M")

        old_odd = Decimal(old_odd.replace(",", ".")) if old_odd != "" or None else None
        odd = Decimal(odd.replace(",", ".")) if odd != "" or None else None
        date = date.replace(hour=heure.hour, minute=heure.minute)

        proba_diff = 1 / old_odd - 1 / odd
        if proba_diff > 0.11:
            golden = "gold"
        else:
            golden = "silver"

        bet = {
            "website": self.WEBSITE,
            "title" : title,
            "sub_title" : sub_title,
            "old_odd" : old_odd,
            "odd" : odd,
            "golden" : golden,
            "sport" : sport,
            "date" : date,
        }
        return bet
        
    def _retrieve_boosted_odds(self) -> list[BoostedOddsObject]:
        """Retrieve the boosted odds from the PSEL website

        Returns:
            list[BoostedOddsObject] : a list with the objects boosted odds, containing the web element + the characteristics of the boosted odd.
        """
        # Get sport classes
        self._get_sports_classes()
        # Find the boosted odds on the page
        try:
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_all_elements_located(
                    (
                        By.XPATH,
                        "//ul[@class='carousel_list']",
                    )
                )
            )
            time.sleep(1)
            important_matches = self.driver.find_elements(
                By.XPATH,
                "//ul[@class='carousel_list']//sports-events-event-card",
            )
            
        except Exception as _:
            print("Error while retrieving the boosted odds")
            important_matches = []

        for important_match in important_matches:
            print("oui")
            try :
                # Retrieve some info
                main_info = important_match.find_element(
                        By.XPATH,
                        ".//a//div//div//scoreboards-scoreboard",
                    ).get_attribute("innerText").split("\n")

                title = main_info[0] + " - " + main_info[2]
                date = main_info[1]

                # find the sport
                class_sport = important_match.find_element(
                        By.XPATH,
                        ".//a//div//div//sports-events-event-info//bcdk-breadcrumb//div//bcdk-breadcrumb-item[1]//span[1]",
                    ).get_attribute("class").split(" ")[-1]
                
                if class_sport in self._sport_dict:
                    sport = self._sport_dict[class_sport]

                # Click on the match
                important_match.click()
                time.sleep(4)

                # find the boosted odds
                try:
                    boosted_odds = self.driver.find_elements(
                        By.TAG_NAME,
                        "sports-boosted-odds-market-card",
                    )
                    for boosted_odd in boosted_odds:
                        # Get the infos of the boosted odd
                        bet = self._get_infos_from_boosted_odd(boosted_odd, title, sport, date)
                        self.list_boosted_odds_objects.append(BoostedOddsObject(boosted_odd, **bet))
                except Exception as e:
                    print("Error while retrieving the boosted odds")
                    print(e)
                    pass
            except Exception as e:
                print(f"error retrieving boosted_odds : {e}")
            
        return self.list_boosted_odds_objects

    def _retrieve_only_good_ones(self) -> None:
        """Retrieve only the good boosted odds that follow some conditions"""
        self.final_list_bet = []
        for bet in self.list_boosted_odds_objects:
            if bet.sport in self.conditions_on_sports[bet.golden]:
                if (
                    (self.conditions_on_sports[bet.golden][bet.sport][0]
                    >= bet.odd) and ((bet.odd - bet.old_odd)/bet.old_odd >= self.conditions_on_sports[bet.golden][bet.sport][1]/100)
                ):
                    bet.print_obj()
                    self.final_list_bet.append(bet)
                else:
                    pass

    def run(self) -> tuple[list[BoostedOddsObject],list[BoostedOddsObject]]:
        """Main function to retrieve the boosted odds from the PSEL website"""
        try:
            self._initiate()
            self._load_page()
            self.all_boosted_odds = self._retrieve_boosted_odds()
            self._retrieve_only_good_ones()
        except Exception as e:
            print(f"An error occurred: {e}")
        return self.all_boosted_odds, self.final_list_bet