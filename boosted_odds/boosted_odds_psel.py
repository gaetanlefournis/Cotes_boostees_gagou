
import datetime
import re
import time
from decimal import Decimal

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from telegram import Bot

from abstract.abstract import AbstractBoostedOdds
from utils.constants import CONDITIONS_ON_SPORTS, URL_BOOSTED_ODDS_PSEL


class BoostedOddsPSEL(AbstractBoostedOdds):
    def __init__(
        self,
        bet_history: list,
        headless=True,
        token_telegram=None,
        chat_id_telegram=None,
    ) -> None:
        self.WEBSITE = 'PSEL'
        self.bet_history = bet_history
        self.headless = headless
        self.url_connexion = URL_BOOSTED_ODDS_PSEL
        self.conditions_on_sports = CONDITIONS_ON_SPORTS["PSEL"]
        self.token = token_telegram
        self.chat_id = chat_id_telegram
        self._sport_dict = None

    def _instantiate(self) -> None:
        self.driver = uc.Chrome(headless=self.headless, use_subprocess=False)

    def load_page(self):
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

    def _find_button_by_text(self, text: str) -> WebElement:
        """Find a button by its text"""
        buttons = self.driver.find_elements(By.TAG_NAME, "button")
        for button in buttons:
            if button.text.lower() == text.lower():
                return button
        return None

    def _close_first_popup(self):
        """Close the first popup"""
        close_button = self._find_button_by_text("close")
        if close_button:
            close_button.click()
            print("First popup closed")
        else:
            print("the button to close the first popup was not found")

    def _accept_cookies(self):
        """Accept the cookies"""
        accept_button = self._find_button_by_text("Continuer sans accepter")
        if accept_button:
            accept_button.click()
            print("Cookies accepted")
        else:
            print("the button to accept the cookies was not found")

    def _get_sports_classes(self):
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
            pattern = r'(\d+(.|,)\d+)\s*(->|→)'
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
            "title" : title,
            "sub_title" : sub_title,
            "old_odd" : old_odd,
            "odd" : odd,
            "golden" : golden,
            "sport" : sport,
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
                        "//div[@class='psel-sport-events psel-webapp-wrapper']",
                    )
                )
            )
            time.sleep(1)
            wrappers = self.driver.find_elements(
                By.XPATH,
                "//div[@class='psel-sport-events psel-webapp-wrapper']",
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
            # Get the infos of the boosted odd
            bet = self._get_infos_from_boosted_odd(boosted_odd)
            list_bet.append(bet)
            
        return list_bet

    def real_bet_to_send(self, list_bet):
        new_list_bet = []
        for bet in list_bet:
            if bet["sport"] in self.conditions_on_sports[bet["golden"]]:
                if (
                    (self.conditions_on_sports[bet["golden"]][bet["sport"]][0]
                    >= bet["odd"]) and ((bet["odd"] - bet["old_odd"])/bet["old_odd"] >= self.conditions_on_sports[bet["golden"]][bet["sport"]][1]/100)
                ):
                    print("\n", bet)
                    new_list_bet.append(bet)
                else:
                    pass
        return new_list_bet

    def _already_in_bet_history(self, bet):
        for bet_h in self.bet_history:
            if bet_h["title"] == bet["title"] and bet_h["sub_title"] == bet["sub_title"] and bet_h["old_odd"] == bet["old_odd"] and bet_h["odd"] == bet["odd"] and bet_h["sport"] == bet["sport"]:
                return True
        return False

    async def send_bet_to_telegram(self, new_list_bet):
        if new_list_bet:
            try:
                bot = Bot(token=self.token)
                for bet in new_list_bet:
                    if not self._already_in_bet_history(bet):
                        message = f"site : {self.WEBSITE}, \nsport: {bet['sport']}, \ntitle : {bet['title']}, \nsubtitle : {bet['sub_title']}, \nold_odd: {bet['old_odd']}, \nnew_odd: {bet['odd']}"
                        await bot.send_message(chat_id=self.chat_id, text=message)
                        self.bet_history.append(bet)
                print("Bets sent successfully!")
            except Exception as e:
                print(f"An error occurred while sending bets: {e}")
        else:
            print("No bet to send")

    async def main(self):
        try:
            self._instantiate()
            self.load_page()
            list_bet = self.retrieve_boosted_odds()
            new_list_bet = self.real_bet_to_send(list_bet)
            await self.send_bet_to_telegram(new_list_bet)
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            self.driver.close()
            self.driver.quit()
        return self.bet_history