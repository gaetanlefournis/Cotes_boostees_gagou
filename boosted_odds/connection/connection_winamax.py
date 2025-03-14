import datetime
import time
import traceback
from decimal import Decimal
from logging import Logger
from typing import List, Tuple

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from utils import constants
from utils.abstract import BettingSiteConnect
from utils.human_behavior import HumanBehavior


class ConnectionWinamax(BettingSiteConnect):
    def __init__(
        self,
        driver: uc.Chrome,
        username: str,
        password: str,
        day: int,
        month: int,
        year: int,
        **kwargs,
    ):
        self.driver = driver
        self.url_connexion = constants.URL_CONNEXION_WINAMAX
        self.username = username
        self.password = password
        self.day = day
        self.month = month
        self.year = year
        self.behavior = HumanBehavior(self.driver)

    def _is_connected(self) -> bool:
        """Check if we are already connected"""
        if "winamax.fr/paris-sportifs" in self.driver.current_url:
            self.driver.get(self.driver.current_url)
        else:
            self.driver.get(constants.URL_CONNEXION_WINAMAX)

        self.behavior.dwell_time()

        try:
            elem = self.driver.find_element(By.ID, "login-link")
            return elem.text.lower() != "se connecter"
        except:
            return True

    def _find_element_by_placeholder(self, placeholder: str) -> WebElement:
        """Find an element by its placeholder"""
        inputs = self.driver.find_elements(By.TAG_NAME, "input")
        for input in inputs:
            if placeholder in input.get_attribute("placeholder"):
                return input
        return None

    def _find_button_by_text(self, text: str) -> WebElement:
        """Find a button by its text"""
        buttons = self.driver.find_elements(By.TAG_NAME, "button")
        for button in buttons:
            if button.text.lower() == text.lower():
                return button
        return None

    def _accept_cookies(self):
        """Accept cookies like the GDPR cookie"""
        accept_button = self._find_button_by_text("tout accepter")
        if accept_button:
            self.behavior.random_click(accept_button)
        return True

    def _login(self):
        """Log in to the website using a human behavior"""
        self.driver.get(self.url_connexion)
        
        time.sleep(4)
        self._accept_cookies()
        self.driver.switch_to.frame("iframe-login")
        self.behavior.dwell_time()
        print("Trying to log in")
        
        username_input = self._find_element_by_placeholder("Email")
        password_input = self._find_element_by_placeholder("Mot de passe")
        self.behavior.random_click(username_input)
        self.behavior.dwell_time()
        self.behavior.human_type(username_input, self.username,rand=False)
        self.behavior.random_click(password_input)
        self.behavior.dwell_time()
        self.behavior.human_type(password_input, self.password,rand=False)
        login_button = self._find_button_by_text("se connecter")
        if login_button:
            self.behavior.random_click(login_button)

        self.behavior.dwell_time()

        self.driver.switch_to.default_content()
        return True

    def _fill_birthday(self):
        """Fill the birthday form that is sometimes needed to connect"""
        if "account/login" not in self.driver.current_url:
            print("Birthday already filled")
        else:
            try:
                self.driver.switch_to.frame("iframe-login")
                self.behavior.dwell_time()
                if (
                    "les informations saisies ne correspondent pas"
                    in self.driver.page_source.lower()
                ):
                    self.driver.get_screenshot_as_file(
                        f"/app/wrong_credentials.png"
                    )
                    print("Wrong credentials")
                    return False
                day_input = self._find_element_by_placeholder("JJ")
                month_input = self._find_element_by_placeholder("MM")
                year_input = self._find_element_by_placeholder("AAAA")
                if day_input is None or month_input is None or year_input is None:
                    print("Unable to find birthday inputs, wrong credentials")
                self.behavior.human_type(day_input, str(self.day), rand=False)
                self.behavior.human_type(month_input, str(self.month), rand=False)
                self.behavior.human_type(year_input, str(self.year), rand=False)

                confirm_button = self._find_button_by_text("se connecter")
                if confirm_button:
                    self.behavior.random_click(confirm_button)
                print("Birthday filled")

                self.behavior.dwell_time()
                self.driver.switch_to.default_content()

                # To check
                view_bet_button = self._find_button_by_text("voir mon pari")
                if view_bet_button:
                    self.behavior.random_click(view_bet_button)
            except:
                # Save screenshort
                print("Unable to fill birthday, cannot log")
                return False
        self.behavior.dwell_time()
        return True

    def close(self, **kwargs):
        """Close the driver"""
        return super().close(**kwargs)

    # ... cookies functions ...

    def run(self):
        """Main function to run the connection"""
        log, birth = False, False
        try:
            log = self._login()
            birth = self._fill_birthday()
        except:
            print("Unable to log to the winamax website")
        if log and birth:
            print(f"Successfully logged to your {self.username} Winamax account")
        else:
            print(f"Unable to log to your {self.username} Winamax account")



