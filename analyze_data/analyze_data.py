import argparse
import os
import shutil
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from utils.constants import LIST_SPORTS
from utils.tools import load_config, save_fig


class AnalyzeDataDB1():
    def __init__(
        self, 
        amount_max, 
        metal,
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
        self.metal = metal
        self.data = None
        self.formatted_dico_amount = None
        self.MIN_THRESHOLD = 1.8
        self.MAX_THRESHOLD = 6.0
        self.INCREMENT_THRESHOLD = 0.1
        if self.metal == "both":
            self.SIZE_TABLE = 50
        elif self.metal == "gold":
            self.SIZE_TABLE = 10
        else:
            self.SIZE_TABLE = 50
        self.MIN_AMOUNT_WON = 50
        self.BASE_BET_AMOUNT = 10
        self.AMOUNT_MAX = amount_max
        self.COEFFICIENT1 = 200
        self.COEFFICIENT2 = 0.1
        self.MIN_PERCENTAGE_CHANGE = 0.0
        self.MAX_PERCENTAGE_CHANGE = 0.25
        self.INCREMENT_PERCENTAGE_CHANGE = 0.01
        self.MIN_RATIO = 0.8
        self.LIST_SPORTS = LIST_SPORTS
        self._instantiate()

    def _instantiate(self):
        self.engine = create_engine(
            f"mysql+mysqlconnector://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_database}"
        )
        self.data = pd.read_sql(f'SELECT * FROM {self.db_table}', self.engine)

    def analyze_results(self) -> Dict:
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
        if self.metal not in ["gold", "silver", "both"]:
            raise ValueError("The golden parameter must be 'gold', 'silver' or 'both'")

        dico_amount = {}
        for sport in self.LIST_SPORTS:
            
            dico_amount[sport] = [None, None, np.NINF, 0, 0]
            df_sport = self.data[self.data['sport'] == sport]

            if self.metal == "both":
                df_sport = df_sport[df_sport['golden'] != "special"]
            else:
                df_sport = df_sport[df_sport['golden'] == self.metal]

            df_sport = df_sport.dropna(subset=['result'])
            
            for threshold in np.arange(self.MIN_THRESHOLD, self.MAX_THRESHOLD, self.INCREMENT_THRESHOLD):
                threshold = np.round(threshold, 2)
                df_threshold = df_sport[df_sport['odd'].astype(float) <= threshold]
                for percentage in np.arange(self.MIN_PERCENTAGE_CHANGE, self.MAX_PERCENTAGE_CHANGE, self.INCREMENT_PERCENTAGE_CHANGE):
                    percentage = np.round(percentage, 2)
                    df_threshold_percentage = df_threshold[(df_threshold['odd'] - df_threshold['old_odd'])/df_threshold['old_odd'] >= percentage]
                    if df_threshold_percentage.shape[0] >= self.SIZE_TABLE:
                        list_amount = []
                        total_amount = 0
                        for _, row in df_threshold_percentage.iterrows():
                            if str(row['result']).lower() == 'gagné':
                                amount = self.BASE_BET_AMOUNT * (float(row['odd']) - 1)
                            elif str(row['result']).lower() == 'perdu': 
                                amount = -self.BASE_BET_AMOUNT
                            else:
                                amount = 0
                            total_amount += amount
                            list_amount.append(total_amount)
                    
                        if list_amount[-1] >= dico_amount[sport][2] and list_amount[-1]/df_threshold_percentage.shape[0] >= self.MIN_RATIO:
                            dico_amount[sport] = [threshold, percentage, list_amount[-1], len(list_amount), np.round(list_amount[-1]/df_threshold_percentage.shape[0], 2)]

        
        self.formatted_dico_amount = {sport: {'threshold': values[0], 'percentage': values[1], 'won': np.round(values[2], 1), 'amount': self._calculate_amount(values[2], values[3]), 'ratio': values[4]} for sport, values in dico_amount.items() if values[0] is not None and values[1] is not None and values[2] >= self.MIN_AMOUNT_WON}

        return self.formatted_dico_amount
    
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

    def plot_results(self):
        """ Plot the results of the analysis """
        for sport, small_dico in self.formatted_dico_amount.items():
            list_amount = []
            total_amount = 0
            fig = plt.figure(figsize=(15, 10))
            df_sport = self.data[self.data['sport'] == sport]

            if self.metal == "both":
                df_sport = df_sport[df_sport['golden'] != "special"]
            else:
                df_sport = df_sport[df_sport['golden'] == self.metal]

            df_sport = df_sport.dropna(subset=['result'])

            df_threshold = df_sport[df_sport['odd'].astype(float) <= small_dico['threshold']]
            df_percentage = df_threshold[(df_threshold['odd'].astype(float) - df_threshold['old_odd'].astype(float))/df_threshold['old_odd'].astype(float) >= small_dico['percentage']]
            
            for _, row in df_percentage.iterrows():
                if str(row['result']).lower() == 'gagné':
                    amount = self.BASE_BET_AMOUNT * (float(row['odd']) - 1)
                elif str(row['result']).lower() == 'perdu': 
                    amount = -self.BASE_BET_AMOUNT
                else:
                    amount = 0
                total_amount += amount
                list_amount.append(total_amount)

            plt.plot(list_amount)
            plt.title(f"Evolution of the amount of money won for {sport}, threshold: {small_dico['threshold']}, percentage: {small_dico['percentage']}, golden: {self.metal}")
            plt.xlabel("Number of bets")
            plt.ylabel("Amount of money won")
            save_fig(fig, f"/home/gagou/Documents/Projet/Cotes_boostees_gagou/results/{self.db_table}/{self.metal}/{sport}_{small_dico['threshold']}.png")
            plt.close()

    def clear_folder(self):
        """ Clear the folder containing the plots before saving the new ones if the folder already exists"""
        if os.path.exists(f"/home/gagou/Documents/Projet/Cotes_boostees_gagou/results/{self.db_table}/{self.metal}/"):
            shutil.rmtree(f"/home/gagou/Documents/Projet/Cotes_boostees_gagou/results/{self.db_table}/{self.metal}/")
        else:
            os.makedirs(f"/home/gagou/Documents/Projet/Cotes_boostees_gagou/results/{self.db_table}/{self.metal}/")
               
    def close_engine(self):
        """ Close the engine """
        self.engine.dispose()
                




