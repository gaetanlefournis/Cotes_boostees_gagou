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
from utils.constants import CONDITIONS_ON_SPORTS, URL_BOOSTED_ODDS_WINAMAX
from utils.tools import find_button_by_text


class RetrieverWinamax(AbstractRetriever):
    """Class to retrieve the boosted odds from winamax. After the connection, the final goal is to return a list of boosted odds that respect the conditions in the constants file."""
    def __init__(
        self,
        driver : uc.Chrome = None,
        **kwargs,
    ) -> None:
        self.WEBSITE = "winamax"
        self.url_connexion = URL_BOOSTED_ODDS_WINAMAX
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
        """Load the page winamax.fr and close the popups and accept the cookies"""
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
        accept_button = find_button_by_text(self.driver, "autoriser")
        if accept_button:
            accept_button.click()
            print("Cookies accepted")
        else:
            print("the button to accept the cookies was not found")

    def _get_sports_classes(self) -> None:
        """Get a dictionary with the sports and their classes. This will be useful to know afterwards the sports of the boosted odds.
        
        Example : 
        {
            "football" : "cmkGtn",
            "tennis" : "hzmxVd",
            "basketball" : "gvPzBl",
        }
        """
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
        if len(text) == 5:
            heure, title, sub_title, old_odd, odd = (
                text[0],
                text[1],
                text[2],
                text[3],
                text[4],
            )
        elif len(text) == 4:
            heure, title, sub_title, odd = (
                text[0],
                text[1],
                text[2],
                text[3],
            )
        else:
            print("Error while getting the boosted odd infos, text length is not 4 or 5")

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

        for boosted_odd in boosted_odds:
            # Get the infos of the boosted odd
            try:
                bet = self._get_infos_from_boosted_odd(boosted_odd)
                self.list_boosted_odds_objects.append(BoostedOddsObject(boosted_odd, **bet))
            except Exception as e:
                print(f"Error while getting info from boosted odd : {e}")
                continue
                
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