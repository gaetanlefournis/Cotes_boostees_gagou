import datetime
import time
from decimal import Decimal

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text
from telegram import Bot

from abstract.abstract import AbstractBoostedOdds
from utils.constants import CONDITIONS_ON_SPORTS, URL_BOOSTED_ODDS_WINAMAX


class BoostedOddsWinamax(AbstractBoostedOdds):
    def __init__(
        self,
        db_database : str,
        db_user : str,
        db_password : str,
        db_host : str,
        db_port : str,
        db_table : str,
        headless : bool = True,
        token_telegram : str = None,
        chat_id_telegram : str = None,
        **kwargs,
    ) -> None:
        self.db_database = db_database
        self.db_user = db_user
        self.db_password = db_password
        self.db_host = db_host
        self.db_port = db_port
        self.db_table = db_table
        self.WEBSITE = 'winamax'
        self.headless = headless
        self.url_connexion = URL_BOOSTED_ODDS_WINAMAX
        self.conditions_on_sports = CONDITIONS_ON_SPORTS["winamax"]
        self.token = token_telegram
        self.chat_id = chat_id_telegram
        self._sport_dict = None
        self.final_list_bet = None

    def _instantiate(self) -> None:
        self.driver = uc.Chrome(headless=self.headless, use_subprocess=False)
        self.engine = create_engine(
            f"mysql+mysqlconnector://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_database}"
        )
        self.session = sessionmaker(bind=self.engine)()

    def load_page(self):
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
        accept_button = self._find_button_by_text("autoriser")
        if accept_button:
            accept_button.click()
            print("Cookies accepted")
        else:
            print("the button to accept the cookies was not found")

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
                
        return list_bet

    def real_bet_to_send(self, list_bet) -> None:
        self.final_list_bet = []
        for bet in list_bet:
            if bet["sport"] in self.conditions_on_sports[bet["golden"]]:
                if (
                    (self.conditions_on_sports[bet["golden"]][bet["sport"]][0]
                    >= bet["odd"]) and ((bet["odd"] - bet["old_odd"])/bet["old_odd"] >= self.conditions_on_sports[bet["golden"]][bet["sport"]][1]/100)
                ):
                    print("\n", bet)
                    self.final_list_bet.append(bet)
                else:
                    pass

    def add_bets_in_db(self) -> None:
        """Every valid bet must be put in the new database"""
        for bet in self.final_list_bet:
            if self._already_in_db(bet):
                continue
            else:
                # Add the bet in the db
                query = text(f"INSERT INTO {self.db_table} (website, sport, title, sub_title, old_odd, odd, golden, statut, date) VALUES (:website, :sport, :title, :sub_title, :old_odd, :odd, :golden, 'PENDING', :date)")
                self.session.execute(query, {"website": bet["website"], "sport": bet["sport"], "title": bet["title"], "sub_title": bet["sub_title"], "old_odd": bet["old_odd"], "odd": bet["odd"], "golden": bet["golden"], "date": bet["date"]})
                self.session.commit()

    def _already_in_db(self, bet) -> bool:
        """Check if the bet is already in the database"""
        query = text(f"SELECT * FROM {self.db_table} WHERE website = :website AND sport = :sport AND title = :title AND sub_title = :sub_title AND old_odd = :old_odd AND odd = :odd AND golden = :golden")
        result = self.session.execute(query, {"website": bet["website"], "sport": bet["sport"], "title": bet["title"], "sub_title": bet["sub_title"], "old_odd": bet["old_odd"], "odd": bet["odd"], "golden": bet["golden"]})
        if result.rowcount > 0:
            return True
        return False

    async def send_bet_to_telegram(self) -> None:
        if self.final_list_bet:
            try:
                bot = Bot(token=self.token)
                for bet in self.final_list_bet:
                    if not self._already_in_db(bet):
                        message = f"site : {self.WEBSITE}, \nsport: {bet['sport']}, \ntitle : {bet['title']}, \nsubtitle : {bet['sub_title']}, \nold_odd: {bet['old_odd']}, \nnew_odd: {bet['odd']}"
                        await bot.send_message(chat_id=self.chat_id, text=message)
                print("Bets sent successfully!")
            except Exception as e:
                print(f"An error occurred while sending bets: {e}")
        else:
            print("No bet to send")

    async def main(self) -> list:
        try:
            self._instantiate()
            self.load_page()
            list_bet = self.retrieve_boosted_odds()
            self.real_bet_to_send(list_bet)
            await self.send_bet_to_telegram()
            self.add_bets_in_db()
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            self.driver.close()
            self.driver.quit()
        return self.final_list_bet