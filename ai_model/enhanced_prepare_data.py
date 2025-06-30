import numpy as np
import pandas as pd
import torch
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from tqdm import tqdm
from transformers import CamembertModel, CamembertTokenizer

from ai_model.mlflow.mlflow import MLFlow


class EnhancedPrepareData:
    """
    Optimized data preparation with key improvements:
    1. Target encoding for sports
    2. Golden-specific odds features
    3. Proper CLS token handling for CamemBERT
    4. French stopwords
    5. Sport-win-rate interaction
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        text_embedding_method: str = 'tfidf',
        use_class_token: bool = False,
        test_size: float = 0.2,
        val_size: float = 0.2,
        chronological_split: bool = True,
        random_state: int = 42,
        batch_size_embedding: int = 128,
        threshold_sports: int = 50,
        only_golden: bool = False,
        without_golden: bool = False,
        **kwargs
    ):
        self.df = df.copy()
        self.text_embedding_method = text_embedding_method
        self.use_class_token = use_class_token
        self.test_size = test_size
        self.val_size = val_size
        self.chronological_split = chronological_split
        self.random_state = random_state
        self.batch_size_embedding = batch_size_embedding
        self.threshold_sports = threshold_sports
        self.only_golden = only_golden
        self.without_golden = without_golden
        self.selected_text_features = None
        self.final_train_df = None
        self.final_val_df = None
        self.final_test_df = None
        
        if self.text_embedding_method == 'camembert':
            self.tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
            self.camembert = CamembertModel.from_pretrained("camembert-base")
            if torch.cuda.is_available():
                self.camembert = self.camembert.to('cuda')

    def _filter_data(self) -> None:
        """Clean and filter the raw data"""
        print("size of the data before filtering:", len(self.df))
        # Initial filtering
        if self.only_golden:
            self.df = self.df[self.df['golden'] == 'gold']

        elif self.without_golden:
            self.df = self.df[self.df['golden'] == 'silver']
        
        self.df = self.df[self.df['result'].isin(['Gagné', 'Perdu'])]
        self.df = self.df.dropna(subset=['old_odd', 'odd', 'sport', 'golden'])
        
        # Convert odds
        self.df['old_odd'] = pd.to_numeric(self.df['old_odd'], errors='coerce')
        self.df['odd'] = pd.to_numeric(self.df['odd'], errors='coerce')
        self.df = self.df.dropna(subset=['old_odd', 'odd'])
        
        # Filter sports
        sport_counts = self.df['sport'].value_counts()
        valid_sports = sport_counts[sport_counts >= self.threshold_sports].index
        self.df = self.df[self.df['sport'].isin(valid_sports)]
        
        self._clean_text_columns()
        self._create_features()

        print("size of the data after filtering:", len(self.df))

    def _clean_text_columns(self) -> None:
        """Clean text with French-specific handling"""
        patterns = ['La Grosse Cote Boostée', 'La Grosse Cote Boostee', 'La Cote boostée de Noël', 'Cote Boostée', 'Cote Boostee', 'Spécial Noël']
        for pattern in patterns:
            self.df['title'] = self.df['title'].str.replace(pattern, '', regex=False)
        
        self.df['description'] = (
            self.df['title'].str.replace(r'\(.*?\)|[^\w\s]', ' ', regex=True) + ' ' +
            self.df['sub_title'].str.replace(r'\(.*?\)|[^\w\s]', ' ', regex=True)
        ).str.strip()
        
        self.df = self.df.drop(columns=['title', 'sub_title'])

    def _create_features(self) -> None:
        """Enhanced feature engineering"""
        # Core odds features
        self.df['odds_ratio'] = self.df['odd'] / self.df['old_odd']
        self.df['odds_diff'] = self.df['odd'] - self.df['old_odd']
        
        # New: Golden-specific features
        self.df['golden_odds_boost'] = self.df['odds_ratio'] * self.df['golden'].eq('gold').astype(int)
        
        
        # Binary and text features
        self.df['odds_percentage_change'] = (
            (self.df['odd'] - self.df['old_odd']) / self.df['old_odd']
        ).fillna(0).clip(lower=-1, upper=1)
        if not self.without_golden:
            self.df['is_golden'] = self.df['golden'].eq('gold').astype(int)
        self.df['desc_length'] = self.df['description'].str.len()
        self.df['word_count'] = self.df['description'].str.split().str.len()

    def _embed_text(self, texts: pd.Series) -> np.ndarray:
        """Improved text embedding with CLS token support"""
        if self.text_embedding_method == 'tfidf':
            return TfidfVectorizer(
                max_features=500,
                ngram_range=(1, 2),
                stop_words='french'
            ).fit_transform(texts).toarray()
        
        elif self.text_embedding_method == 'count':
            return CountVectorizer(
                max_features=500,
                ngram_range=(1, 2),
                stop_words='french'
            ).fit_transform(texts).toarray()
        
        elif self.text_embedding_method == 'camembert':
            embeddings = []
            for i in tqdm(range(0, len(texts), self.batch_size_embedding)):
                inputs = self.tokenizer(
                    texts.iloc[i:i+self.batch_size_embedding].tolist(),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128
                )
                if torch.cuda.is_available():
                    inputs = {k: v.to('cuda') for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.camembert(**inputs)
                    if self.use_class_token:
                        batch_emb = outputs.last_hidden_state[:, 0, :]  # CLS token
                    else:
                        batch_emb = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
                    embeddings.append(batch_emb.cpu())
            embeddings = torch.cat(embeddings).numpy()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return embeddings

    def _build_preprocessor(self, train_df: pd.DataFrame) -> ColumnTransformer:
        """Optimized preprocessing pipeline with:
        - Target encoding for sports
        - SelectKBest for text features
        - Proper golden feature handling
        """
        # Numeric features (odds + derived)
        numeric_features = [
            'old_odd', 'odd', 'odds_ratio', 'odds_diff',
            'odds_percentage_change'
        ]
        # Add golden-specific features
        if not self.without_golden:
            numeric_features.append('golden_odds_boost')
            numeric_features.append('is_golden')

        
        # Add text length features if available
        if 'desc_length' in train_df.columns:
            numeric_features.extend(['desc_length', 'word_count'])
        
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical features (sport only)
        categorical_transformer = Pipeline([
            ('target_enc', TargetEncoder()),  # From category_encoders
            ('scaler', StandardScaler())
        ])
        
        # Text processing pipeline
        text_transformer = Pipeline([
            ('embed', FunctionTransformer(self._embed_text, validate=False)),
            ('select', SelectKBest(f_classif, k=100))  # Top 100 text features
        ])
        
        transformers = [
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, ['sport']),
            ('text', text_transformer, 'description')
        ]
        
        return ColumnTransformer(transformers, remainder='drop')

    def _prepare_features(self, df: pd.DataFrame, y: pd.Series = None) -> torch.Tensor:
        """Convert DataFrame to tensor after preprocessing"""
        if df is None or df.empty:
            return None
            
        if not hasattr(self, 'preprocessor'):
            self.preprocessor = self._build_preprocessor(df)
            X = self.preprocessor.fit_transform(df, y)

            if hasattr(self.preprocessor, 'transformers_'):
                text_transformer = self.preprocessor.named_transformers_['text']
                if hasattr(text_transformer, 'named_steps'):
                    self.selected_text_features = text_transformer.named_steps['select'].get_support()
        else:
            X = self.preprocessor.transform(df)
        
        return torch.FloatTensor(X) if X is not None else None
    
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
        shuffled_df = self.df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)

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
    
    def log_parameters(self, mlflow: MLFlow) -> None:
        """Log parameters for MLFLOW"""
        mlflow.log_params("text_embedding_method", self.text_embedding_method)
        mlflow.log_params("use_class_token", self.use_class_token)
        mlflow.log_params("test_size", self.test_size)
        mlflow.log_params("val_size", self.val_size)
        mlflow.log_params("chronological_split", self.chronological_split)
        mlflow.log_params("random_state", self.random_state)
        mlflow.log_params("batch_size_embedding", self.batch_size_embedding)
        mlflow.log_params("threshold_sports", self.threshold_sports)
        mlflow.log_params("only_golden", self.only_golden)
        mlflow.log_params("without_golden", self.without_golden)

    # Keep your exact __call__ method as provided
    def __call__(self) -> tuple[tuple[torch.Tensor, torch.Tensor], 
                          tuple[torch.Tensor, torch.Tensor] | None, 
                          tuple[torch.Tensor, torch.Tensor]]:
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

        # Prepare targets
        y_train = self.final_train_df["result"].map({'Gagné': 1, 'Perdu': 0})
        y_test = self.final_test_df["result"].map({'Gagné': 1, 'Perdu': 0})
        y_val = self.final_val_df["result"].map({'Gagné': 1, 'Perdu': 0}) if self.final_val_df is not None else None

        # Prepare features
        X_train = self._prepare_features(self.final_train_df, y_train)
        X_test = self._prepare_features(self.final_test_df, None)
        X_val = self._prepare_features(self.final_val_df, None) if self.final_val_df is not None else None

        # Convert to tensors
        y_train = torch.LongTensor(y_train.values)
        y_test = torch.LongTensor(y_test.values)
        y_val = torch.LongTensor(y_val.values) if y_val is not None else None

        return (X_train, y_train), (X_test, y_test), (X_val, y_val)