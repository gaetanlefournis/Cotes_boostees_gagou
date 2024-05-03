import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text

from utils.constants import LIST_SPORTS


# Description: Add sports to the database
class AddSportsAutomatic():

    def __init__(
        self,
        user: str,
        password: str,
        host: str,
        port: str,
        database: str,
        table: str,
        **kwargs,
    ):
        self.db_user = user
        self.db_password = password
        self.db_host = host
        self.db_port = port
        self.db_database = database
        self.db_table = table
        self.engine = None
        self._instantiate()

    def _instantiate(self) -> None:
        """Instantiate the database engine"""
        self.engine = create_engine(
            f"mysql+mysqlconnector://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_database}"
        )
        self.session = sessionmaker(bind=self.engine)()

    def add_sports_automatic(self):
        """Add sports to the database if the sport appears in the title or subtitle"""
        SELECT_QUERY = f"SELECT * FROM {self.db_table}"
        df = pd.read_sql(SELECT_QUERY, self.engine)
        for index, row in df.iterrows():
            for sport in LIST_SPORTS:
                if sport.lower() in row['title'].lower() or sport.lower() in row['sub_title'].lower():
                    query = text(f"UPDATE {self.db_table} SET sport = '{sport}' WHERE id = {row['ID']}")
                    self.session.execute(query)
                    self.session.commit()
                else:
                    pass
    
    def add_sports_with_word(self, word : str = "", sport : str = ""):
        """Add sports to the database thanks to a word we give, that corresponds to a sport"""
        SELECT_QUERY = f"SELECT * FROM {self.db_table}"
        df = pd.read_sql(SELECT_QUERY, self.engine)
        for index, row in df.iterrows():
            if word in row['title'] or word in row['sub_title']:
                query = text(f"UPDATE {self.db_table} SET sport = '{sport}' WHERE id = {row['ID']}")
                self.session.execute(query)
                self.session.commit()
            else:
                pass

    def close(self):
        """Close the connection to the database"""
        self.engine.dispose()

    def __call__(self, word : str = "", sport : str = ""):
        if word != "" and sport != "":
            self.add_sports_with_word(word, sport)
        else:
            self.add_sports_automatic()
        self.close()

    


