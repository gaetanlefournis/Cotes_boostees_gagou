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