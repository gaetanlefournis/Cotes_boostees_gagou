import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text

from utils.constants import DICO_SPORTS


# Description: Add sports to the database
class AddSportsManual():

    def __init__(
        self,
        db_user: str,
        db_password: str,
        db_host: str,
        db_port: str,
        db_database: str,
        table_name: str,
        **kwargs,
    ):
        self.db_user = db_user
        self.db_password = db_password
        self.db_host = db_host
        self.db_port = db_port
        self.db_database = db_database
        self.db_table = table_name
        self.engine = None
        self._instantiate()

    def _instantiate(self) -> None:
        """Instantiate the database engine"""
        self.engine = create_engine(
            f"mysql+mysqlconnector://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_database}"
        )
        self.session = sessionmaker(bind=self.engine)()

    
    def add_sports(self):
        """Add sports to the database thanks to a word we give, that corresponds to a sport"""
        SELECT_QUERY = f"SELECT * FROM {self.db_table} WHERE sport IS NULL OR sport = ''"
        condition = True
        while condition:
            df = pd.read_sql(SELECT_QUERY, self.engine)
            if df.empty:
                print("No more lines to fill with a sport.")
                break
            for _, row in df.iterrows():
                title = row['title']
                sub_title = row['sub_title']
                print(f'\n\n\n\ntitle : {title}\nsub_title : {sub_title}\n')
                letter = str(input("what is the sport ? Enter a letter (q to quit): "))
                if letter not in DICO_SPORTS.keys() and letter != "q":
                    print("The letter is not recognized. Please enter a letter from the list.")
                    continue
                elif letter == "q":
                    condition = False
                    break
                else:
                    sport = DICO_SPORTS[letter]
                    query = text(f"UPDATE {self.db_table} SET sport = '{sport}' WHERE id = {row['ID']}")
                    self.session.execute(query)
                    self.session.commit()

    def close(self):
        """Close the connection to the database"""
        self.engine.dispose()

    def __call__(self):
        self.add_sports()
        self.close()

    