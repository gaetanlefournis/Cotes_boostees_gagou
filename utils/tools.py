import os
import time

import dotenv
import undetected_chromedriver as uc
import yaml
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from undetected_chromedriver import Chrome


def substitute_placeholders(value, env: dict):
    if isinstance(value, str) and value.startswith("$"):
        if env is None:
            return os.environ.get(value[2:-1])  # Fetch value from environment variable
        return env.get(value[2:-1])  # Fetch value from environment variable
    if isinstance(value, dict):
        return {k: substitute_placeholders(v, env) for k, v in value.items()}
    if isinstance(value, list):
        return [substitute_placeholders(v, env) for v in value]
    return value

def load_config(yaml_path: str, env_var_path: str = None):
    """Convert a yaml file to a dict

    Args:
        yaml (str): The path to the yaml file

    Returns:
        dict: The dict corresponding to the yaml file
    """
    env = None
    if env_var_path is not None:
        env = dotenv.dotenv_values(env_var_path)
    with open(yaml_path) as file:
        config = yaml.safe_load(file)

    config = substitute_placeholders(config, env)

    return config

def save_fig(fig, path: str):
    """Save a figure to a file, if the path doesn't exist, create it"""
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    fig.savefig(path)


######################### TOOLS WINAMAX #########################

def click_on_odd_button(boosted_odd: WebElement) -> None:
    """Click on the button of a boosted odd

    Args:
        boosted_odd (webdriver.remote.webelement.WebElement): The boosted odd.
        golden (str): "yes" if it's a golden odd, "no" if it's a silver odd, "other" if it's a special odd.

    Returns:
        success (bool): True if the click was successful, False otherwise.
    """

    try:
        (
            boosted_odd.find_element(
                By.XPATH,
                ".//div//div[1]//div//div[2]//div//div//div[2]//div[2]//div[2]",
            )
        ).click()
        return True
    except Exception as e:
        pass

    try:
        (
            boosted_odd.find_element(
                By.XPATH, ".//div//div[1]//div//div[2]//div[2]//div[2]//div[2]//div[2]"
            )
        ).click()
        return True
    except:
        pass

    return False

def close_right_panel(driver: uc.Chrome) -> None:
    """Close the right panel

    Args:
        driver (uc.Chrome): The driver

    Returns:
        None
    """
    i = 0
    while i < 5:
        try:
            right_panel = driver.find_element(By.XPATH, "//div[@data-testid='rightColumn']")
            if "ton panier est vide" in right_panel.text.lower():
                return True
        except:
            return True
        try :
            
            right_panel.find_element(By.XPATH, ".//div//div//div//div[1]//div[3]//div[3]//div//div//button[2]").click() #bouton fermer
            time.sleep(1)
            continue
        except :
            pass
        try :
            #click on the dustbin
            dust_bin_pannel = right_panel.find_element(By.XPATH, "//div[@data-testid='rightColumn']")
            dust_bin_pannel.find_element(By.XPATH, ".//div//div//div//div[1]//div[1]//div[1]//div//svg").click()
            time.sleep(1)
            
        except:
            pass
        i += 1
    return False

def bet_on_Winamax(driver: uc.Chrome, betting_stake: float) -> None:
    """Bet on Winamax

    Args:
        betting_stake (float): The amount of money to bet

    Returns:
        success (bool): True if the bet was successful, False otherwise.
    """

    try:
        # Get the right window
        right_window = driver.find_element(
            By.XPATH, "//div[@data-testid='sticky-wrap']"
        )

        # Get the input to bet
        input_bet = right_window.find_element(By.XPATH, ".//input")
    except:
        return False

    try:
        # Enter the amount of money to bet
        input_bet.clear()
        input_bet.send_keys(float(betting_stake))
        
        # Click on the button to bet
        right_window.find_element(By.XPATH, ".//button").click()
        time.sleep(5)

    except:
        return False

    time.sleep(1)
    try:
        WebDriverWait(driver, 10).until(
            EC.text_to_be_present_in_element(
                (By.XPATH, "//div[@data-testid='sticky-wrap']"),
                "pari validé",
            )
        )
        return True
    
    except Exception as e:
        return False

def find_button_by_text(driver : Chrome, text: str) -> WebElement:
        """Find a button by its text"""
        buttons = driver.find_elements(By.TAG_NAME, "button")
        for button in buttons:
            if button.text.lower() == text.lower():
                return button
        return None