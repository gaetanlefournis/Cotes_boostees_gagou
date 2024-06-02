from abc import ABC, abstractmethod


class AbstractRetriever(ABC):
    
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

class AbstractBoostedOdds(ABC):
    
    @abstractmethod
    def _instantiate(self):
        pass

    @abstractmethod
    def load_page(self):
        pass

    @abstractmethod
    def retrieve_boosted_odds(self):
        pass

    @abstractmethod
    def real_bet_to_send(self, list_bet):
        pass

    @abstractmethod
    async def send_bet_to_telegram(self, new_list_bet):
        pass

    @abstractmethod
    async def main(self):
        pass