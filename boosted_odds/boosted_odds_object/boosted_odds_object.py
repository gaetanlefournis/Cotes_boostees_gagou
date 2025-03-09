from dataclasses import dataclass

from selenium.webdriver.remote.webelement import WebElement


@dataclass
class BoostedOddsObject:
    """This object is used to store the WebElement of a boosted odd + its informations that will be put in the database, and will be useful to interact with the database, and later to bet on the odd."""
    boosted_odd : WebElement
    website : str
    sport : str
    title : str
    sub_title : str
    date : str
    old_odd : float
    odd : float
    golden : str
    dictionary : dict = None

    def __post_init__(self) -> None:
        """Create the dictionary of the object"""
        self.dictionary = {"boosted_odd" : self.boosted_odd, "website": self.website, "sport": self.sport, "title": self.title, "sub_title": self.sub_title, "date": self.date, "old_odd": self.old_odd, "odd": self.odd, "golden": self.golden}

    def print_obj(self) -> None:
        """Print the object"""
        print("\n", self.dictionary)
