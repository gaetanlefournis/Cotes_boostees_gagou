import time
from datetime import datetime
from typing import Any

import pandas as pd
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text

from utils.abstract import AbstractRetrieverBetBoosted
from utils.constants import URL_BET_BOOSTED_BETCLIC


class RetrieverBetBoostedBetclic(AbstractRetrieverBetBoosted):
    def __init__(
            self,
            db_database: str,
            db_user: str,
            db_password: str,
            db_host: str,
            db_port: str,
            table: str,
            global_retrieve: bool = False,
            **kwargs,
    ):
        self.headless = False
        self.user_agent = "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US) AppleWebKit/532.0 (KHTML, like Gecko) Chrome/3.0.195.38 Safari/532.0"
        self.FIRST_ELEMENT = 1
        self.LAST_ELEMENT_WITHOUT_GLOBAL = 500
        self.LAST_ELEMENT = 5000
        self.url_bet_boosted = URL_BET_BOOSTED_BETCLIC
        self.db_database = db_database
        self.db_user = db_user
        self.db_password = db_password
        self.db_host = db_host
        self.db_port = db_port
        self.db_table = table
        self.global_retrieve = global_retrieve
        self.driver = None
        self.engine = None
        self.session = None

    def _instantiate(self) -> None:
        """Instantiate the driver and the database engine"""
                # Driver setup
        options = uc.ChromeOptions()

        if not self.headless:  # Ensure headless is False
            print("Debug Mode: Headless Disabled")
            options.add_argument("--disable-blink-features=AutomationControlled")  # Prevent detection
            options.add_argument("--disable-gpu")
            options.add_argument("--force-device-scale-factor=1")
        else:
            options.add_argument("--headless=new")  # Use "new" for better compatibility

        options.add_argument(f"user-agent={self.user_agent}")
        options.add_argument("--use_subprocess=True")
        options.add_argument("--no-sandbox")
        options.add_argument("--start-maximized")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--remote-debugging-port=9222")

        # Anti-detection improvements
        options.add_argument("--disable-blink-features=AutomationControlled")  # Prevent bot detection
        options.add_argument("--disable-infobars")  # Disable "Chrome is being controlled by automated software"
        options.add_argument("--disable-extensions")  # Disable extensions
        options.add_argument("--disable-software-rasterizer")  # Disabling the software renderer
        options.add_argument("--disable-features=VizDisplayCompositor")  # Prevents UI blocking

        # Initialize Chrome
        self.driver = uc.Chrome(options=options)
        self.engine = create_engine(
            f"mysql+mysqlconnector://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_database}"
        )
        self.session = sessionmaker(bind=self.engine)()


    def load_page(self):
        """Load the page wepari.fr and accept the cookies"""
        self.driver.get(self.url_bet_boosted)

        try : 
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            time.sleep(5)
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

    def _accept_cookies(self):
        """Accept the cookies"""
        accept_button = self._find_button_by_text("autoriser")
        if accept_button:
            accept_button.click()
            print("Cookies accepted")
        else:
            print("the button to accept the cookies was not found")

    def retrieve_all(self):
        """Retrieve by changing the page everytime"""
        is_last_page = False
        all_odds = self.driver.find_element(By.XPATH, "//button[@value='all']")
        all_odds.click()
        time.sleep(3)
        while not is_last_page:
            button_next_page = self.driver.find_element(By.XPATH, "//button[@aria-label='Next page']")
            self.retrieve_all_data()
            is_last_page = button_next_page.get_attribute("disabled") is not None
            if not is_last_page:
                button_next_page.click()
                time.sleep(2)

    def retrieve_all_data(self):
        """Loop through the main element to retrieve all the data"""
        # Retrieve the list of elements on which we will loop, being careful that there are several pages
        list_elements = self.driver.find_elements(By.XPATH, "//div[@class='v-table__wrapper']//table//tbody//tr")

        data_table = []
        # Loop through the elements
        for element in list_elements:
            data_row = self.retrieve_data(element)

            # Check if the data is already in the database. If yes, we change the column result of this line in the database
            query = text(f"SELECT * FROM {self.db_table} WHERE title = :title AND sub_title = :sub_title AND old_odd = :old_odd AND odd = :odd AND date = :date")
            result = self.session.execute(query, data_row).fetchall()
            if result and result[0][7] != data_row["result"]:
                print(f"Data already in the database : {result}")
                print(f"old result : {result[0][7]}")
                print(f"new result : {data_row['result']}")
                # modify the result in the database
                query = text(f"UPDATE {self.db_table} SET result = :result WHERE title = :title AND sub_title = :sub_title AND old_odd = :old_odd AND odd = :odd AND date = :date")
                self.session.execute(query, data_row)
                self.session.commit()
            elif not result:
                data_table.append(data_row)

        # Fill the database
        df = pd.DataFrame(data_table)
        df.to_sql(self.db_table, self.engine, if_exists="append", index=False)


    def retrieve_data(self, element : WebElement) -> dict:
        """Retrieve the data from the element"""
        list_td = element.find_elements(By.XPATH, ".//td")
        text = []
        for td in list_td:
            text.append(td.get_attribute("innerText"))

        # Retrieve the obvious data
        sub_title = text[2]
        odd = float(text[4])
        if text[3] != '':
            old_odd = float(text[3])
        else:
            old_odd = 0
        date = text[0].split(" ")[0]
        date = datetime.strptime(date, "%Y-%m-%d").date()
        title = text[1]
        golden = "gold"

        # Retrieve the result attribute to the last number
        if float(text[6]) > 1:
            result = "Gagné"
        elif float(text[6]) < -1:
            result = "Perdu"
        else:
            result = "En cours"

        # No way to guess the sport in the data...
        return {
            "sport": None,
            "title": title,
            "sub_title": sub_title,
            "old_odd": old_odd,
            "odd": odd,
            "golden": golden,
            "result": result,
            "date": date,
        }

    def close(self):
        """Close the driver"""
        self.driver.close()
        self.driver.quit()
        self.session.close()

    def __call__(self, *args: Any, **kwds: Any):
        try:
            self._instantiate()
            self.load_page()
            self.retrieve_all()
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            self.close()





            
        

