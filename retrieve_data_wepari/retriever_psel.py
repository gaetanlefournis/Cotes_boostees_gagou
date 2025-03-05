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

from abstract.abstract import AbstractRetriever
from utils.constants import URL_WEPARI_PSEL


class RetrieverPSEL(AbstractRetriever):
    def __init__(
            self,
            db_database: str,
            db_user: str,
            db_password: str,
            db_host: str,
            db_port: str,
            table: str,
            global_retrieve: bool = True,
            **kwargs,
    ):
        self.headless = False
        self.user_agent = "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US) AppleWebKit/532.0 (KHTML, like Gecko) Chrome/3.0.195.38 Safari/532.0"
        self.FIRST_ELEMENT = 1
        self.LAST_ELEMENT_WITHOUT_GLOBAL = 500
        self.LAST_ELEMENT = 5000
        self.url_wepari = URL_WEPARI_PSEL
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
        self._instantiate()

    def _instantiate(self) -> None:
        """Instantiate the driver and the database engine"""
        self.driver = uc.Chrome(headless=self.headless, use_subprocess=False, user_agent=self.user_agent)
        self.engine = create_engine(
            f"mysql+mysqlconnector://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_database}"
        )
        self.session = sessionmaker(bind=self.engine)()


    def load_page(self):
        """Load the page wepari.fr and accept the cookies"""
        self.driver.get(self.url_wepari)

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

    def retrieve_all_data(self):
        """Loop through the main element to retrieve all the data"""
        # Retrieve the main body
        main_element = self.driver.find_element(By.TAG_NAME, "body").find_element(By.TAG_NAME, "table").find_element(By.TAG_NAME, "table").find_element(By.TAG_NAME, "tbody")

        # Retrieve the list of elements on which we will loop
        list_elements = main_element.find_elements(By.TAG_NAME, "tr")
        if not self.global_retrieve:
            list_elements = list_elements[self.FIRST_ELEMENT:self.LAST_ELEMENT_WITHOUT_GLOBAL]
        else : 
            list_elements = list_elements[self.FIRST_ELEMENT:self.LAST_ELEMENT]
        list_elements.reverse()

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
        list_td = element.find_elements(By.TAG_NAME, "td")

        # Retrieve the obvious data
        sub_title = list_td[0].text
        odds = list_td[1].text.split("\n")
        odd = odds[0]
        old_odd = odds[1]
        date = list_td[3].text.split(" ")[0]
        date = datetime.strptime(date, "%d-%m-%Y").date()
        title = list_td[4].text

        # Retrieve the golden attribute thanks to the color
        if list_td[0].get_attribute("bgcolor") == "yellow":
            golden = "gold"
        else:
            golden = "silver"

        # Retrieve the result attribute to the color
        if list_td[1].get_attribute("bgcolor") == "green":
            result = "Gagné"
        elif list_td[1].get_attribute("bgcolor") == "orange":
            result = "Perdu"
        elif list_td[6].text == "":
            result = "En cours"
        else:
            result = "Annulé"
            
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
        self.load_page()
        self.retrieve_all_data()
        self.close()

