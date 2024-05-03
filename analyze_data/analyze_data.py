import argparse
from typing import Dict

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from utils.constants import LIST_SPORTS
from utils.tools import load_config


class AnalyzeDataDB1():
    def __init__(
        self, 
        amount_max, 
        user,
        password, 
        host,
        port, 
        database,
        table,
        **kwargs
    ) -> None:
        self.db_user = user
        self.db_password = password
        self.db_host = host
        self.db_port = port
        self.db_database = database
        self.db_table = table
        self.data = None
        self.MIN_THRESHOLD = 2.0
        self.MAX_THRESHOLD = 6.0
        self.INCREMENT_THRESHOLD = 0.1
        self.SIZE_TABLE = 50
        self.MIN_AMOUNT_WON = 50
        self.BASE_BET_AMOUNT = 10
        self.AMOUNT_MAX = amount_max
        self.COEFFICIENT1 = 200
        self.COEFFICIENT2 = 0.1
        self._instantiate()

    def _instantiate(self):
        self.engine = create_engine(
            f"mysql+mysqlconnector://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_database}"
        )
        self.data = pd.read_sql(f'SELECT * FROM {self.db_table}', self.engine)

    def analyze_results(self, list_sport: list = LIST_SPORTS, golden: str = "both") -> Dict:
        """
        Analyze the results of the bets and save the plots.

        Parameters:
        - df: DataFrame containing the betting data.
        - list_sport: List of sports to analyze.
        - golden: Type of golden bets to consider ('gold', 'silver', or 'both').

        Returns:
        - DataFrame containing the formatted analysis results.
        """
        # Validate input parameters
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError("Input 'df' must be a DataFrame")
        if not isinstance(list_sport, list) or len(list_sport) == 0:
            raise ValueError("Input 'list_sport' must be a non empty list of sports")
        if golden not in ["gold", "silver", "both"]:
            raise ValueError("The golden parameter must be 'gold', 'silver' or 'both'")

        dico_amount = {}
        for sport in list_sport:
            
            dico_amount[sport] = [None, 0, 0]
            df_sport = self.data[self.data['sport'] == sport]

            if golden == "both":
                df_sport = df_sport[df_sport['golden'] != "special"]
            else:
                df_sport = df_sport[df_sport['golden'] == golden]

            df_sport = df_sport.dropna(subset=['result'])
            
            for threshold in np.arange(self.MIN_THRESHOLD, self.MAX_THRESHOLD, self.INCREMENT_THRESHOLD):
                threshold = np.round(threshold, 2)
                df_threshold = df_sport[df_sport['odd'].astype(float) <= threshold]

                if df_threshold.shape[0] >= self.SIZE_TABLE:
                    list_amount = []
                    total_amount = 0
                    for _, row in df_threshold.iterrows():
                        if str(row['result']).lower() == 'gagné':
                            amount = self.BASE_BET_AMOUNT * (float(row['odd']) - 1)
                        elif str(row['result']).lower() == 'perdu': 
                            amount = -self.BASE_BET_AMOUNT
                        else:
                            amount = 0
                        total_amount += amount
                        list_amount.append(total_amount)
                
                    if list_amount[-1] >= dico_amount[sport][1]:
                        dico_amount[sport] = [threshold, list_amount[-1], len(list_amount)]
        
        formatted_dico_amount = {sport: {'threshold': values[0], 'won': np.round(values[1], 1), 'amount': self._calculate_amount(values[1], values[2])} for sport, values in dico_amount.items() if values[0] is not None and values[1] >= self.MIN_AMOUNT_WON}

        return formatted_dico_amount
    
    def _calculate_amount(self, money_won : float, nb_bets : int) -> float:
        """
        Calculate the amount to bet based on the money won and the number of bets. The greater the amount won per bet, the bigger the amount to bet. The more bets, the bigger the amount to bet.

        Parameters:
        - amount_max: Maximum amount allowed for betting.
        - money_won: Amount of money won.
        - nb_bets: Number of bets.

        Returns:
        - float: Amount to bet.
        """
        return max(0.1, np.round((np.sqrt(self.AMOUNT_MAX) - np.log(1 + self.COEFFICIENT1/nb_bets) - np.log(1 + self.COEFFICIENT2*nb_bets/money_won))**2, 2))



                




