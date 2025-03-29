from abc import ABC, abstractmethod

import undetected_chromedriver as uc

from boosted_odds.boosted_odds_object.boosted_odds_object import \
    BoostedOddsObject


class AbstractRetrieverWePari(ABC):
    
    @abstractmethod
    def _instantiate(self):
        pass

    @abstractmethod
    def load_page(self):
        pass

    @abstractmethod
    def retrieve_all_data(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def __call__(self, *args, **kwds):
        pass

class AbstractRetrieverBetBoosted(ABC):
    
    @abstractmethod
    def _instantiate(self):
        pass

    @abstractmethod
    def load_page(self):
        pass

    @abstractmethod
    def retrieve_all_data(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def __call__(self, *args, **kwds):
        pass

class AbstractRetriever(ABC):
    
    @abstractmethod
    def _initiate(self):
        pass

    @abstractmethod
    def _load_page(self):
        pass

    @abstractmethod
    def _retrieve_boosted_odds(self):
        pass

    @abstractmethod
    def _get_infos_from_boosted_odd(self):
        pass

    @abstractmethod
    def _retrieve_only_good_ones(self):
        pass

    @abstractmethod
    def run(self):
        pass

class AbstractBettor(ABC):

    @abstractmethod
    def _initiate(self):
        pass

    @abstractmethod
    def _connection_to_website(self):
        pass

    @abstractmethod
    def _bet_on_boosted_odd(self, boosted_odd : BoostedOddsObject):
        pass

    @abstractmethod
    def run(self, list_boosted_odds : list[BoostedOddsObject]):
        pass

class AbstractBetAutomation(ABC):

    @abstractmethod
    def _instantiate(self):
        pass

    @abstractmethod
    def connection_to_website(self, list_bets):
        pass

    @abstractmethod
    def retrieve_pending_bets(self):
        pass

    @abstractmethod
    def bet_on_bet(self, boosted_odd, bet):
        pass

    @abstractmethod
    def change_bet_statut(self, bet):
        pass

    @abstractmethod
    def main(self):
        pass

class BettingSiteConnect(ABC):
    """Abstract class that will be implemented by the connection to the betting websites"""

    @abstractmethod
    def __init__(
        self,
        driver: uc.Chrome,
        username: str,
        password: str,
        url_connexion: str,
        *args,
        **kwargs,
    ):
        """Initialize the connection to the betting website

        Args:
            driver (uc.Chrome): The driver of the browser
            username (str): username of the betting website
            password (str): password of the betting website
            url_connexion (str): url of the connexion page of the betting website
        """
        pass

    @abstractmethod
    def _is_connected(self, *args, **kwargs) -> bool:
        """Check if the connection to the betting website is successful

        Returns:
            bool: True if the connection is successful, False otherwise
        """
        pass

    @abstractmethod
    def run(self, *args, **kwargs):
        """run the connection to the betting website process"""
        pass

    @abstractmethod
    def close(self, *args, **kwargs):
        """Close the connection to the betting website"""
        pass