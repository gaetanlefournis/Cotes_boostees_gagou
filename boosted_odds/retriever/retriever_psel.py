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
from utils.constants import CONDITIONS_ON_SPORTS, URL_BOOSTED_ODDS_PSEL
from utils.tools import find_button_by_text


class RetrieverPSEL(AbstractRetriever):
    """Class to retrieve the boosted odds from PSEL. After the connection, the final goal is to return a list of boosted odds that respect the conditions in the constants file."""
    def __init__(
        self,
        driver : uc.Chrome = None,
        **kwargs,
    ) -> None:
        self.WEBSITE = "PSEL"
        self.url_connexion = URL_BOOSTED_ODDS_PSEL
        self.conditions_on_sports = CONDITIONS_ON_SPORTS[self.WEBSITE]
        self._sport_dict = None
        self.final_list_bet = None
        self.list_boosted_odds_objects = []
        self.all_boosted_odds = None
        self.driver = driver

    def _initiate(self) -> None:
        """Create the driver if it doesn't exist yet"""
        if not self.driver:
            self.driver = uc.Chrome(headless=True, use_subprocess=False)

    def _load_page(self) -> None:
        """Load the page parionssport.fdj.fr and close the popups and accept the cookies"""
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
        """Accept the cookies"""
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
                    (By.XPATH, "//psel-ept-sports")
                )
            )
            left = self.driver.find_element(
                By.XPATH, "//psel-ept-sports"
            ).find_element(By.XPATH, ".//ul[1]")

            # Get the list of the sports
            sports = left.find_elements(By.XPATH, "./*")
            _sport_dict = {}
            for sport in sports:
                try:
                    _sport_dict[sport.text] = (
                        sport.find_element(By.XPATH, ".//button//span[1]//span")
                        .get_attribute("class")
                        .split(" ")
                    )
                except:
                    pass
            # For each sport key keep only elements in the list that are unique amongst all the lists of the dict
            new_dict = {}
            for key, value in _sport_dict.items():
                new_dict[value[0]] = key
            self._sport_dict = new_dict
                
        except Exception as e:
            print("Error while getting the sports classes")
            print(e)

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
        text = boosted_odd.text.split("\n")
        if len(text) == 6:
            title, heure, sub_title, odd = (
                text[2] + " " + text[0],
                text[1],
                text[3] + " " + text[4],
                text[5],
            )
        elif len(text) == 7:
            title, heure, sub_title, odd = (
                text[2] + " " + text[0],
                text[1],
                text[4] + " " + text[5],
                text[6],
            )
        elif len(text) == 9:
            title, heure, sub_title, odd = (
                text[2] + text[3] + text[4] + " " + text[0],
                text[1],
                text[5] + " " + text[7],
                text[8],
            )
        elif len(text) == 10:
            title, heure, sub_title, odd = (
                text[3] + text[4] + text[5] + " " + text[0],
                text[1],
                text[6] + " " + text[8],
                text[9],
            )
        else:
            title, heure, sub_title, odd = (
                text[2] + " " + text [3] + " " + text[4] + " " + text[0],
                text[1],
                text[5] + " " + text[6],
                text[7],
            )

        sport = boosted_odd.find_element(
            By.XPATH, ".//a//div[1]//span//span[1]"
        ).get_attribute("class")
        
        
        classes_sport = sport.split(" ")
        for class_sport in classes_sport:
            if class_sport in self._sport_dict:
                sport = self._sport_dict[class_sport]
                break
        jour = heure.split()[0]
        heure = heure.split()[-1]
        if "Demain" in jour:
            date = datetime.datetime.now() + datetime.timedelta(days=1)
        elif "Hier" in jour:
            date = datetime.datetime.now() - datetime.timedelta(days=1)
        else:
            date = datetime.datetime.now()
        heure = datetime.datetime.strptime(heure.split()[-1], "%Hh%M")

        def extract_number_before_arrow(text):
            pattern = r'(\d+(?:[.,]\d+)?)\s*(?=->|→)'
            match = re.search(pattern, text)
            return  match.group(1) if match else None

        old_odd = extract_number_before_arrow(sub_title)
        old_odd = Decimal(old_odd.replace(",", ".")) if old_odd != "" or None else None
        odd = Decimal(odd.replace(",", ".")) if odd != "" or None else None
        date = date.replace(hour=heure.hour, minute=heure.minute)

        proba_diff = 1 / old_odd - 1 / odd
        if proba_diff > 0.11:
            golden = "gold"
        else:
            golden = "silver"

        bet = {
            "website": "PSEL",
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
                        "//div[@class='psel-sport-events']",
                    )
                )
            )
            time.sleep(1)
            wrappers = self.driver.find_elements(
                By.XPATH,
                "//div[@class='psel-sport-events']",
            )
            
            boosted_odds = wrappers[0].find_elements(
                By.XPATH,
                "./*",
            )[1:]
            
        except Exception as _:
            print("Error while retrieving the boosted odds")
            boosted_odds = []

        list_bet = []
        for boosted_odd in boosted_odds:
            if "Réessayez" in boosted_odd.text:
                print("No boosted odds")
                return []
            else :
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