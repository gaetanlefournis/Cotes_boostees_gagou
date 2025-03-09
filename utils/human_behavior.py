import random
import time

from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys


class HumanBehavior:
    def __init__(self, driver):
        self.driver = driver
        self.actions = ActionChains(driver)
    
    def alt_tab(self):
        # press alt tab
        self.actions.key_down(Keys.ALT).send_keys(Keys.TAB).key_up(Keys.ALT).perform()

    def _random_delay(self, short_delay=0.5, long_delay=2.0):
        """Introduce a random delay."""
        time.sleep(random.uniform(short_delay, long_delay))

    def graduate_scroll_to_bottom(self,driver : webdriver.Chrome):
        """Gradually scroll to the bottom of the page."""
        scroll_height = driver.execute_script("return document.body.scrollHeight;")
        current_scroll = 0
        while current_scroll < scroll_height:
            driver.execute_script(f"window.scrollTo(0, {current_scroll});")
            self._random_delay(0.1, 0.3)
            current_scroll += random.randint(100, 500)
            
    def graduate_scroll_to_top(self,driver : webdriver.Chrome):
        """Gradually scroll to the bottom of the page."""
        scroll_height = driver.execute_script("return document.body.scrollHeight;")
        current_scroll = scroll_height
        while current_scroll > 0:
            driver.execute_script(f"window.scrollTo(0, {current_scroll});")
            self._random_delay(0.1, 0.3)
            current_scroll -= random.randint(100, 500)
            
    def random_click(self, element):
        """Click on an element at a random position."""
        w, h = element.size["width"], element.size["height"]

        # If width or height is 0, just click the element.
        if w == 0 or h == 0:
            self.actions.click(element).perform()
            self._random_delay()
            return

        # Combine move and click into a single action chain
        self.actions.move_to_element_with_offset(element, 0, 0).click(element).perform()
        self._random_delay()

    def human_type(self, element, text, rand=True):
        """Type text into an element with randomized intervals."""
        for char in text:
            if random.random() < 0.05 and rand:  # 5% chance to make a typo
                element.send_keys(random.choice("asdfghjkl"))
                self._random_delay(0.1, 0.3)
                element.send_keys(Keys.BACKSPACE)
            element.send_keys(char)
            self._random_delay(0.15, 0.20)

    def curved_mouse_movement(self, element):
        """Move to an element using a non-linear path."""
        mid_point = (
            element.location["x"] + random.randint(-10, 10),
            element.location["y"] + random.randint(-10, 10),
        )
        self.actions.move_by_offset(*mid_point).perform()
        self._random_delay()
        self.actions.move_to_element(element).perform()

    def gradual_scroll(self, element):
        """Scroll to an element gradually and try to place it in the middle of the page."""

        # Get element's position
        target_y = element.location["y"]

        # Calculate the distance needed to position the element in the center of the viewport
        viewport_height = self.driver.execute_script("return window.innerHeight;")
        element_height = element.size["height"]
        center_offset = (viewport_height - element_height) / 2

        # Adjust the target position to center the element
        target_y = target_y - center_offset

        # Get current scroll position
        current_y = self.driver.execute_script("return window.pageYOffset;")
        step = 50 if target_y > current_y else -50

        while (step > 0 and current_y < target_y) or (
            step < 0 and current_y > target_y
        ):
            # Scroll
            self.driver.execute_script(f"window.scrollBy(0, {step});")

            # Add a delay to simulate smooth scrolling
            self._random_delay(0.01, 0.03)

            # Update the current position
            current_y += step

            # Adjust step as we approach the target to avoid overshooting
            if abs(target_y - current_y) < abs(step):
                step = target_y - current_y

    def dwell_time(self, short_delay=1.0, long_delay=3.0):
        """Simulate a human reading or examining the page."""
        self._random_delay(short_delay, long_delay)

    def navigate_using_tab(self, times=1):
        """Use the Tab key to navigate."""
        for _ in range(times):
            self.actions.send_keys(Keys.TAB).perform()
            self._random_delay()

    def navigate_using_arrows(self, direction=Keys.ARROW_RIGHT, times=1):
        """Use arrow keys to navigate."""
        for _ in range(times):
            self.actions.send_keys(direction).perform()
            self._random_delay()

    def resize_viewport(self):
        """Randomly resize the browser window."""
        width = self.driver.get_window_size()["width"]
        height = self.driver.get_window_size()["height"]
        self.driver.set_window_size(
            width + random.randint(-10, 10), height + random.randint(-10, 10)
        )

    def hover_nearby(self, element):
        """Hover around the target element."""
        nearby_elements = self.driver.find_elements(By.XPATH, "//*")
        nearby_element = random.choice(nearby_elements)
        self.actions.move_to_element(nearby_element).perform()
        self._random_delay()
        self.actions.move_to_element(element).perform()

    def clipboard_action(self, element, text):
        """Simulate copying and pasting a text into an element."""
        self.driver.execute_script(f"navigator.clipboard.writeText('{text}');")
        element.send_keys(Keys.CONTROL, "v")
