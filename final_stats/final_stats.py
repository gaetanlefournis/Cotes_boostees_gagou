import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text


class FinalStats():
    def __init__(
        self,
        db_user,
        db_password,
        db_host,
        db_port,
        db_database,
        db_table,
        **kwargs
    ) -> None:
        self.db_user = db_user
        self.db_password = db_password
        self.db_host = db_host
        self.db_port = db_port
        self.db_database = db_database
        self.db_table = db_table
        self.data = None
        self.engine = None
        self.session = None
        self.dico_websites = {"winamax" : {"gold" : 10, "silver" : 25}, "PSEL" : {"gold" : 10, "silver" : 25}, "unibet" : {"gold" : 10, "silver" : 25}}
        self._instantiate()

    def _instantiate(self):
        """"""
        self.engine = create_engine(
            f"mysql+mysqlconnector://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_database}"
        )
        self.session = sessionmaker(bind=self.engine)()
        self.data = pd.read_sql(f'SELECT * FROM {self.db_table}', self.engine)

    def analyze_results(self) -> float:
        """
        """
        total_amount_PSEL = 0
        total_amount_winamax = 0
        total_amount_unibet = 0
        for _, row in self.data.iterrows():
            try:
                if row['result'].lower() == 'gagné':
                    amount = self.dico_websites[row['website']][row['golden']] * (float(row['odd']) - 1)
                elif row['result'].lower() == 'perdu':
                    amount = -self.dico_websites[row['website']][row['golden']]
                else:
                    amount = 0
                if row['website'] == 'winamax':
                    total_amount_winamax += amount
                elif row['website'] == 'PSEL':
                    total_amount_PSEL += amount
                else:
                    total_amount_unibet += amount
            except Exception as e:
                print(f"Error processing row: {row}")
                print(e)
                continue
        return total_amount_winamax, total_amount_PSEL, total_amount_unibet
    
    def update_db(self):
        """Update the database thanks to the other tables and the results of the bets."""
        for _, row in self.data.iterrows():
            if row["result"] is None:
                website = row["website"]

                # Search for the odd in the table of that website and update the result
                query = text(f"SELECT * FROM {website} WHERE sport = :sport AND title = :title AND result IS NOT NULL")
                boosted_odd = self.session.execute(query, {"sport": row["sport"], "title": row["title"]})

                # Fetch the result(s) from boosted_odd (this returns a Result object)
                result = boosted_odd.fetchone()  # Use fetchone() to get a single row
                if result:
                    print(result)
                    # Update the general database with the result
                    query = text(f"UPDATE {self.db_table} SET result = :result WHERE website = :website AND sport = :sport AND title = :title AND sub_title = :sub_title")
                    self.session.execute(query, {
                        "result": result[7],  # Access the data using dictionary-style access
                        "website": website,
                        "sport": row["sport"],
                        "title": row["title"],
                        "sub_title": row["sub_title"]
                    })
                    self.session.commit()
               
    def close_engine(self):
        """ Close the engine """
        self.session.close()
        self.engine.dispose()
                




