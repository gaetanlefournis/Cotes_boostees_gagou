import datetime
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
from utils.constants import CONDITIONS_ON_SPORTS, URL_BOOSTED_ODDS_UNIBET
from utils.tools import find_button_by_text


class RetrieverUnibet(AbstractRetriever):
    """Class to retrieve the boosted odds from unibet. After the connection, the final goal is to return a list of boosted odds that respect the conditions in the constants file."""
    def __init__(
        self,
        driver : uc.Chrome = None,
        **kwargs,
    ) -> None:
        self.WEBSITE = "unibet"
        self.url_connexion = URL_BOOSTED_ODDS_UNIBET
        self.conditions_on_sports = CONDITIONS_ON_SPORTS[self.WEBSITE]
        self._sport_dict = None
        self.final_list_bet = None
        self.list_boosted_odds_objects = []
        self.all_boosted_odds = None
        self.driver = driver

    def _initiate(self):
        """Instantiate the object if there are. Nothing for the moment"""
        pass

    def _load_page(self) -> None:
        """Load the page unibet.fr and close the popups and accept the cookies"""
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
        """Close the first popup which is the -18 popup"""
        close_button = find_button_by_text(self.driver, "close")
        if close_button:
            close_button.click()
            print("First popup closed")
        else:
            print("the button to close the first popup was not found")

    def _accept_cookies(self) -> None:
        """Close the cookies popup by accepting the cookies"""
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
                    (By.XPATH, "//aside[@class='leftside']")
                )
            )
            left = self.driver.find_element(
                By.XPATH, "//aside[@id='leftside']"
            ).find_element(By.XPATH, ".//nav//div[3]//div//div//ul")

            # Get the list of the sports
            sports = left.find_elements(By.XPATH, "./*")
            _sport_dict = {}
            for sport in sports[1:]:
                try:
                    sport_text = sport.find_element(By.XPATH, ".//a//span").get_attribute("innerText")
                    _sport_dict[sport_text] = (
                        sport.find_element(By.XPATH, ".//a//img")
                        .get_attribute("src")
                    )
                except:
                    pass

            # invert the values
            new_dict = {}
            for key, value in _sport_dict.items():
                new_dict[value] = key
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
        if len(text) == 7:
            heure, title, sub_title, old_odd, odd = (
                text[2],
                text[0] + " " + text[3],
                text[4],
                text[5],
                text[6],
            )
        elif len(text) == 8:
            heure, title, sub_title, old_odd, odd = (
                text[2],
                text[0] + " " + text[3],
                text[4] + " " + text[5],
                text[6],
                text[7],
            )
        else:
            print("length not expected, check unibet")
            raise Exception 

        # Find the sport with the logo
        sport = boosted_odd.find_element(
            By.XPATH, ".//div[1]//div//img"
        ).get_attribute("src")
        
        
        sport = self._sport_dict[sport]

        # Get the date
        jour = heure.split()[0]
        heure = heure.split()[-1]
        if "Demain" in jour:
            date = datetime.datetime.now() + datetime.timedelta(days=1)
        elif "Hier" in jour:
            date = datetime.datetime.now() - datetime.timedelta(days=1)
        else:
            date = datetime.datetime.now()

        heure = datetime.datetime.strptime(heure.split()[-1], "%Hh%M")

        # Process the odds
        old_odd = Decimal(old_odd.replace(",", ".")) if old_odd != "" or None else None
        odd = Decimal(odd.replace(",", ".")) if odd != "" or None else None

        # Check if the odd is gold
        proba_diff = 1 / old_odd - 1 / odd
        if proba_diff > 0.11 and odd < 4:
            golden = "gold"
        else:
            golden = "silver"

        # Create the bet
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
        """Retrieve the boosted odds from the Winamax website

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
                        "//section[@id='view-main-container']",
                    )
                )
            )
            time.sleep(1)
            boosted_odds = self.driver.find_element(
                By.XPATH,
                "//section[@id='view-main-container']//section//div//div//div",
            )
            boosted_odds = boosted_odds.find_elements(
                By.XPATH,
                "./*",
            )
            
        except Exception as _:
            print("Error while retrieving the boosted odds")
            boosted_odds = []

        for boosted_odd in boosted_odds:
            # Get the infos of the boosted odd
            bet = self._get_infos_from_boosted_odd(boosted_odd)
            self.list_boosted_odds_objects.append(BoostedOddsObject(boosted_odd, **bet))
                
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

    def run(self) -> tuple[list[BoostedOddsObject], list[BoostedOddsObject]]:
        """Main function to retrieve the boosted odds from the Winamax website"""
        try:
            self._initiate()
            self._load_page()
            self.all_boosted_odds = self._retrieve_boosted_odds()
            self._retrieve_only_good_ones()
        except Exception as e:
            print(f"An error occurred: {e}")
        return self.all_boosted_odds, self.final_list_bet