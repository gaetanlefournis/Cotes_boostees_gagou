
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text


class Database():
    """
    Class to interact with the database
    """
    def __init__(
        self, 
        db_database : str,
        db_user : str,
        db_password : str,
        db_host : str,
        db_port : str,
        db_table : str
    ) -> None:
        self.db_database = db_database
        self.db_user = db_user
        self.db_password = db_password
        self.db_host = db_host
        self.db_port = db_port
        self.db_table = db_table
        self.engine = None
        self.session = None
        self._connect()

    def _connect(self) -> None:
        """
        Connect to the database
        """
        self.engine = create_engine(
            f"mysql+mysqlconnector://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_database}"
        )
        self.session = sessionmaker(bind=self.engine)()

    def insert(self, data : dict) -> None:
        """
        Insert data in the database
        """
        query = text(f"INSERT INTO {self.db_table} (website, sport, title, sub_title, old_odd, odd, golden, statut, date) VALUES (:website, :sport, :title, :sub_title, :old_odd, :odd, :golden, 'PENDING', :date)")
        self.session.execute(query, {"website": data["website"], "sport": data["sport"], "title": data["title"], "sub_title": data["sub_title"], "old_odd": data["old_odd"], "odd": data["odd"], "golden": data["golden"], "date": data["date"]})
        self.session.commit()

    def already_in_db(self, data : dict) -> bool:
        """
        Check if the data is already in the database
        """
        query = text(f"SELECT * FROM {self.db_table} WHERE website = :website AND sport = :sport AND title = :title AND sub_title = :sub_title AND old_odd = :old_odd AND odd = :odd AND golden = :golden")
        result = self.session.execute(query, {"website": data["website"], "sport": data["sport"], "title": data["title"], "sub_title": data["sub_title"], "old_odd": data["old_odd"], "odd": data["odd"], "golden": data["golden"]})
        if result.rowcount > 0:
            return True
        return False
    
    def already_bet_statut(self, data : dict) -> bool:
        """Check if the bet is already with a 'BET' statut"""
        query = text(f"SELECT * FROM {self.db_table} WHERE website = :website AND sport = :sport AND title = :title AND sub_title = :sub_title AND old_odd = :old_odd AND odd = :odd AND golden = :golden AND statut = :statut")
        result = self.session.execute(query, {"website": data["website"], "sport": data["sport"], "title": data["title"], "sub_title": data["sub_title"], "old_odd": data["old_odd"], "odd": data["odd"], "golden": data["golden"], "statut": "BET"})
        if result.rowcount > 0:
            return True
        return False

    def update_bet_statut(self, data : dict) -> None:
        """
        Update data in the database
        """
        query = text(f"UPDATE {self.db_table} SET statut = 'BETTED' WHERE website = :website AND sport = :sport AND title = :title AND sub_title = :sub_title AND old_odd = :old_odd AND odd = :odd")
        self.session.execute(query, {'website': data['website'], 'sport': data['sport'], 'title': data['title'], 'sub_title': data['sub_title'], 'old_odd': data['old_odd'], 'odd': data['odd']})
        self.session.commit()

    def close(self):
        """Close the engine"""
        self.session.close()
        self.engine.dispose()