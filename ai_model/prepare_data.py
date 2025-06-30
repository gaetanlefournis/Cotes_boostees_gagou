import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm import tqdm
from transformers import CamembertModel, CamembertTokenizer

from utils.constants import LIST_SPORTS


class PrepareData:
    """
    Class to prepare the data for AI model training.
    Handles text embedding, categorical encoding, and numerical feature scaling.
    
    Args:
        df (pd.DataFrame): Input dataframe with raw data
        text_embedding_method (str): Method for text embedding ('tfidf', 'camembert', 'count')
        include_cotes (bool): Whether to include betting odds as features
        include_temporal (bool): Whether to include temporal features
    """
    def __init__(
        self,
        df: pd.DataFrame,
        features_columns: dict = {'text': ["description"], 'num': ["old_odd", "odd", "ratio_odds"], 'categorical': ["sport", "golden"]},
        target_columns: list = ['result'],
        text_embedding_method: str = 'tfidf',
        use_class_token: bool = False,
        test_size: float = 0.2,
        val_size: float = 0.2,
        chronological_split: bool = True,
        random_state_split: int = 42,
        random_state_visualization: int = 42,
        batch_size_embedding: int = 128,
        threshold_sports: int = 50,
        odds_ratio_method: str = 'percentage',
        only_golden: bool = False,
        one_hot_encode_sports: bool = True,
        **kwargs
    ):
        self.df = df
        self.categorical_columns = features_columns.get('categorical', [])
        self.numeric_columns = features_columns.get('num', [])
        self.text_columns = features_columns.get('text', [])
        self.target = target_columns
        self.text_embedding_method = text_embedding_method
        self.use_class_token = use_class_token
        self.test_size = test_size
        self.val_size = val_size if not chronological_split else None
        self.chronological_split = chronological_split
        self.random_state_split = random_state_split
        self.random_state_visualization = random_state_visualization
        self.batch_size_embedding = batch_size_embedding
        self.threshold_sports = threshold_sports
        self.odds_ratio_method = odds_ratio_method
        self.only_golden = only_golden
        self.one_hot_encode_sports = one_hot_encode_sports
        self.preprocessor = None
        self.final_test_df = None
        self.final_train_df = None
        self.final_val_df = None
        
        if self.text_embedding_method == 'camembert':
            self.tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
            self.camembert = CamembertModel.from_pretrained("camembert-base")
            if torch.cuda.is_available():
                self.camembert = self.camembert.to('cuda')

    def _filter_data(self) -> None:
        """
        Remove every row with a result that is not 'Gagné' or 'Perdu'.
        Remove every row with a "old_odd" or "odd" that is not a number.
        Remove every row with a "sport" that is not in the list of sports.
        Remove evry row with a "sport" that has less than threshold_sports occurrences.
        Remove every row with a "golden" that is not in ['silver', 'gold'].
        Remove every occurence of "Cote Boostée" or "Cote Boostee" or "La Grosse Cote Boostée" or "La Grosse Cote Boostee" or "La Cote boostée de Noël", "Spécial Noël" in the "title" column.
        Remove the row if there is "?" in the "title" or "sub_title" column.
        Replace ":" by "" in the "title" or "sub_title" column.
        Remove the sport from the "title" column if it appears in the "title" column.
        Remove everything in parenthesis in the "sub_title" and "title" column.
        Remove the date column because we sort thanks to the ID column.
        Create a column "ratio_odds" that is the ratio of "old_odd" and "odd".

        Join the title and sub_title columns into a single "description" column.
        """
        NUMBER_SAMPLES_VISUALIZATION = 10

        print(f"Initial data shape: {self.df.shape}")

        # If only_golden, then we keep only the rows with golden = 'gold'
        if self.only_golden:
            self.df = self.df[self.df['golden'] == 'gold']
            print(f"Filtered to only golden rows: {self.df.shape[0]} rows")

        # Remove unclean rows
        self.df = self.df[self.df['result'].isin(['Gagné', 'Perdu'])]
        print(f"After filtering result: {self.df.shape[0]} rows")
        self.df = self.df[self.df['old_odd'].apply(lambda x: x is not None)]
        print(f"After filtering old_odd: {self.df.shape[0]} rows")
        self.df = self.df[self.df['odd'].apply(lambda x: x is not None)]
        print(f"After filtering odd: {self.df.shape[0]} rows")
        self.df = self.df[self.df['sport'].isin(LIST_SPORTS)]
        print(f"After filtering sport: {self.df.shape[0]} rows")
        self.df = self.df.groupby('sport').filter(lambda x: len(x) >= self.threshold_sports)
        print(f"After filtering sports with less than {self.threshold_sports} occurrences: {self.df.shape[0]} rows")
        self.df = self.df[self.df['golden'].isin(['silver', 'gold'])]
        print(f"After filtering golden: {self.df.shape[0]} rows")
        self.df = self.df[~self.df['title'].str.contains(r'\?', na=False)]
        self.df = self.df[~self.df['sub_title'].str.contains(r'\?', na=False)]
        print(f"After filtering title and sub_title with '?': {self.df.shape[0]} rows")

        # Remove the possible occurences of what is common in the titles (important to start with the longer ones because the small # ones can be part of the longer ones)
        self.df['title'] = self.df['title'].str.replace('La Cote boostée de Noël', '', regex=False)
        self.df['title'] = self.df['title'].str.replace('La Cote boostee de Noël', '', regex=False)
        self.df['title'] = self.df['title'].str.replace('La Grosse Cote Boostée', '', regex=False)
        self.df['title'] = self.df['title'].str.replace('La Grosse Cote Boostee', '', regex=False)
        self.df['title'] = self.df['title'].str.replace('Cote Boostée', '', regex=False)
        self.df['title'] = self.df['title'].str.replace('Cote Boostee', '', regex=False)
        self.df['title'] = self.df['title'].str.replace('Spécial Noël', '', regex=False)
        self.df['sub_title'] = self.df['sub_title'].str.replace(':', '', regex=False)
        self.df['title'] = self.df['title'].str.replace(':', '', regex=False)
        self.df['title'] = self.df.apply(lambda row: row['title'].replace(row['sport'], '').strip(), axis=1)
        self.df['sub_title'] = self.df['sub_title'].str.replace(r'\(.*?\)', '', regex=True)
        self.df['title'] = self.df['title'].str.replace(r'\(.*?\)', '', regex=True)
        self.df.drop(columns=['date'], inplace=True, errors='ignore')

        # Update on old_odd and odd columns
        self.df['old_odd'] = pd.to_numeric(self.df['old_odd'], errors='coerce')
        self.df['odd'] = pd.to_numeric(self.df['odd'], errors='coerce')
        self.df.dropna(subset=['old_odd', 'odd'], inplace=True)

        if self.odds_ratio_method == "percentage":
            self.df['ratio_odds'] = (self.df['odd'] - self.df['old_odd']) / self.df['old_odd']
        elif self.odds_ratio_method == "ratio":
            self.df['ratio_odds'] = self.df['old_odd'] / self.df['odd']

        # Create description column
        self.df['description'] = self.df['title'] + ' ' + self.df['sub_title']

        # Remove unnecessary columns
        self.df.drop(columns=['title', 'sub_title'], inplace=True, errors='ignore')

        # look at the rows that miss at least 1 value
        missing_rows = self.df[self.df.isnull().any(axis=1)]
        if not missing_rows.empty:
            print("Rows with missing values:")
            print(missing_rows)

        # print 10 random rows of the dataframe
        print("Sample of filtered data:")
        print(self.df.sample(n=NUMBER_SAMPLES_VISUALIZATION, random_state=self.random_state_visualization))

    def _embed_text_camembert(self, texts : pd.Series) -> torch.Tensor:
        """Embed text using Camembert model with batch processing"""
        embeddings = []

        for i in tqdm(range(0, len(texts), self.batch_size_embedding)):
            batch = texts[i:i+self.batch_size_embedding]
            inputs = self.tokenizer(
                batch.tolist(), 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=128
            ).to('cuda' if torch.cuda.is_available() else 'cpu')
            
            with torch.no_grad():
                outputs = self.camembert(**inputs)
                hidden_states = outputs.last_hidden_state
                
                if self.use_class_token:
                    # Combine CLS token and mean pooling
                    batch_embeddings = torch.cat([
                        hidden_states[:, 0, :],  # CLS token
                        hidden_states.mean(dim=1)  # Mean pooled
                    ], dim=1)
                else:
                    batch_embeddings = hidden_states.mean(dim=1)  # Just mean pooling
                    
                embeddings.append(batch_embeddings.cpu())  # Move to CPU immediately
            
        final_embeddings = torch.cat(embeddings, dim=0)
        
        return final_embeddings

    def _chronological_split(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the data chronologically without shuffling
        - Train: lowest ID
        - Test: highest ID
        """
        # Sort by ascending ID
        self.df = self.df.sort_values('ID', ascending=True)

        # Calculate the index to split the data
        cutoff_idx = int(len(self.df) * (1 - self.test_size))

        # Split into train/test
        train_df = self.df.iloc[:cutoff_idx]
        test_df = self.df.iloc[cutoff_idx:]
        
        return train_df, test_df
    
    def _random_split(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the data randomly into train and test sets
        """
        # Shuffle the DataFrame
        shuffled_df = self.df.sample(frac=1, random_state=self.random_state_split).reset_index(drop=True)

        # Calculate the index to split the data
        cutoff_idx_test = int(len(shuffled_df) * (1 - self.test_size))

        # Split into train/test/val
        train_val_df = shuffled_df.iloc[:cutoff_idx_test]
        test_df = shuffled_df.iloc[cutoff_idx_test:]

        # Calculate the index for validation split if val_size is provided
        if self.val_size is not None:
            cutoff_idx_val = int(len(train_val_df) * (1 - self.val_size))
            train_df = train_val_df.iloc[:cutoff_idx_val]
            val_df = train_val_df.iloc[cutoff_idx_val:]
        else:
            train_df = train_val_df
            val_df = None


        return train_df, test_df, val_df

    def _prepare_features(self, df : pd.DataFrame) -> torch.Tensor:
        """
        Prepare features for a specific subset (train or test)
        """
        if df is None or df.empty:
            return None
        
        # Create transformers
        transformers = []

        # Add categorical features
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_columns))

        # Add numeric features
        if len(self.numeric_columns) > 0:
            transformers.append(('num', StandardScaler(), self.numeric_columns))

        # Add text embedding
        if self.text_embedding_method == 'tfidf':
            transformers.append(('text', TfidfVectorizer(max_features=1000), self.text_columns))
        elif self.text_embedding_method == 'camembert':
            pass
        
        # Fit preprocessor only on training data
        if not hasattr(self, 'preprocessor_fitted'):
            self.preprocessor = ColumnTransformer(transformers, remainder='drop')
            X = self.preprocessor.fit_transform(df)
            self.preprocessor_fitted = True
        else:
            X = self.preprocessor.transform(df)
        
        # Process text features if Camembert
        if self.text_embedding_method == 'camembert':
            text_features = self._embed_text_camembert(df[self.text_columns[0]])
            if X is not None:
                X = np.hstack([X.toarray() if hasattr(X, "toarray") else X, text_features])
            else:
                X = text_features

        return torch.FloatTensor(X)

    def __call__(self) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        """Main method to prepare data with chronological split"""
        # Filter data first
        self._filter_data()
        
        # Split chronologique
        if self.chronological_split and self.val_size is None:
            self.final_train_df, self.final_test_df = self._chronological_split()
        elif self.chronological_split and self.val_size is not None:
            raise ValueError("Chronological split is True, but val_size is not None. Please set chronological_split to False if you want to use val_size.")
        else:
            self.final_train_df, self.final_val_df, self.final_test_df = self._random_split()

        # Prepare features
        X_train = self._prepare_features(self.final_train_df)
        X_test = self._prepare_features(self.final_test_df)
        X_val = self._prepare_features(self.final_val_df)

        # Prepare targets
        y_train = self.final_train_df["result"].map({'Gagné': 1, 'Perdu': 0})
        y_test = self.final_test_df["result"].map({'Gagné': 1, 'Perdu': 0})
        y_val = self.final_val_df["result"].map({'Gagné': 1, 'Perdu': 0}) if self.final_val_df is not None else None

        y_train = torch.LongTensor(y_train.values)  # .values accesses the numpy array
        y_test = torch.LongTensor(y_test.values)
        y_val = torch.LongTensor(y_val.values) if y_val is not None else None

        return (X_train, y_train), (X_test, y_test), (X_val, y_val)