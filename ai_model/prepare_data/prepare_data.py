import re
from typing import Optional

import numpy as np
import pandas as pd
import spacy
import torch
from category_encoders import TargetEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from stop_words import get_stop_words
from tqdm import tqdm
from transformers import CamembertModel, CamembertTokenizer

from ai_model.mlflow_perso.mlflow_perso import MLFlow
from utils.constants import PATTERNS_LESS, PATTERNS_MORE, PATTERNS_TO_REMOVE


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """French text preprocessor for sports betting data."""
    def __init__(self, lowercase=True, lemmatize=True):
        self.lowercase = lowercase
        self.lemmatize = lemmatize

        # Load French stopwords
        self.stopwords = set(get_stop_words('french'))
        
        # Keep critical terms that should not be lemmatized
        self.stopwords -= {'ou', 'et', 'plus', 'moins', 'mi-temps', 'mi temps', 
                          '1ère', '2ème', 'première', 'seconde'}

        # Add more words to exclude
        self.stopwords |= {'match', 'dans', 'de', 'la', 'le', 'un', 'une'}

        # Load spaCy model
        try:
            self.nlp = spacy.load('fr_core_news_sm', disable=['parser', 'ner'])
        except OSError:
            raise ImportError(
                "The french spaCy model 'fr_core_news_sm' is not installed. "
                "Run : `python -m spacy download fr_core_news_sm`"
            )

    def _normalize_text(self, text: str) -> str:
        """Clean and normalize French text."""
        if self.lowercase:
            text = text.lower()
        # Delete accents and special characters
        text = re.sub(r'[éèêë]', 'e', text)
        text = re.sub(r'[àâä]', 'a', text)
        text = re.sub(r'[îï]', 'i', text)
        text = re.sub(r'[ôö]', 'o', text)
        text = re.sub(r'[ùûü]', 'u', text)
        return re.sub(r'[^a-z0-9\s]', ' ', text)

    def _lemmatize_text(self, text: str) -> str:
        """Lemmatize French text while preserving critical terms."""
        doc = self.nlp(text)
        return ' '.join([
            token.lemma_ if token.text not in {'ou', 'et', 'plus', 'moins'}
            else token.text
            for token in doc
            if token.text not in self.stopwords
        ])

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.apply(self._normalize_text)
        if self.lemmatize:
            X = X.apply(self._lemmatize_text)
        return X

class EnhancedPrepareData:
    """Complete data preparation for sports betting."""
    def __init__(
        self,
        df: pd.DataFrame,
        text_embedding_method: str = 'tfidf',
        use_class_token: bool = False,
        test_size: float = 0.2,
        val_size: Optional[float] = None,
        chronological_split: bool = True,
        random_state: int = 42,
        batch_size_embedding: int = 32,
        threshold_sports: int = 100,
        only_golden: bool = False,
        without_golden: bool = False,
        with_feature_is_golden: bool = True,
        use_text: bool = True,
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
        self.with_feature_is_golden = with_feature_is_golden
        self.use_text = use_text

        self.final_train_df = None
        self.final_val_df = None
        self.final_test_df = None

        self.X_train = None
        self.X_test = None
        self.X_val = None
        self.y_train = None
        self.y_test = None
        self.y_val = None

        if self.text_embedding_method in ('camembert', 'hybrid'):
            self.tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
            self.camembert = CamembertModel.from_pretrained("camembert-base")
            if torch.cuda.is_available():
                self.camembert = self.camembert.to('cuda')

    def _clean_initial_text(self):
        """Clean the title and sub_title columns from the start."""
        # Remove promotional patterns
        for col in ['title', 'sub_title']:
            # Put everything in lower case
            self.df[col] = self.df[col].str.lower()

            # remove the sport if it appears
            self.df[col] = self.df.apply(lambda row: row[col].replace(str(row['sport']).lower(), ''), axis=1)
            for pattern in PATTERNS_TO_REMOVE:
                self.df[col] = self.df[col].str.replace(pattern, '', regex=False)
            
            # Delete accents and special characters
            self.df[col] = self.df[col].str.normalize('NFKD')\
                                       .str.encode('ascii', errors='ignore')\
                                       .str.decode('utf-8')
            
            # Replace unwanted characters and everything inside parenthesis
            self.df[col] = self.df[col].str.replace(r'\(.*?\)', '', regex=True)
            self.df[col] = self.df[col].str.replace(r'[^\w\s]', ' ', regex=True)

    def _filter_data(self) -> None:
        """Initial data filtering."""
        print(f"Initial size: {len(self.df)}")

        self._clean_initial_text()

        # remove the date column if it exists
        if 'date' in self.df.columns:
            self.df = self.df.drop(columns=['date'])

        # Filter golden/silver
        if self.only_golden:
            self.df = self.df[self.df['golden'] == 'gold']
        elif self.without_golden:
            self.df = self.df[self.df['golden'] == 'silver']

        # Filter results and missing values
        self.df = self.df[self.df['result'].isin(['Gagné', 'Perdu'])]
        self.df = self.df.dropna(subset=['old_odd', 'odd', 'sport', 'golden'])

        # Convert odds
        self.df['old_odd'] = pd.to_numeric(self.df['old_odd'], errors='coerce')
        self.df['odd'] = pd.to_numeric(self.df['odd'], errors='coerce')

        # Filter rare sports
        sport_counts = self.df['sport'].value_counts()
        valid_sports = sport_counts[sport_counts >= self.threshold_sports].index
        self.df = self.df[self.df['sport'].isin(valid_sports)]

        # Create features
        self._create_features()
        print(f"Final size: {len(self.df)}")

    def _create_features(self) -> None:
        """Create advanced features."""
        # Odds features
        self.df['odds_ratio'] = self.df['odd'] / self.df['old_odd']
        self.df['odds_diff'] = self.df['odd'] - self.df['old_odd']
        self.df['odds_percentage_change'] = (
            (self.df['odd'] - self.df['old_odd']) / self.df['old_odd']
        ).fillna(0).clip(-1, 1)
        
        # Features golden
        self.df['golden_odds_boost'] = self.df['odds_ratio'] * self.df['golden'].eq('gold').astype(int)
        if not self.without_golden:
            self.df['is_golden'] = self.df['golden'].eq('gold').astype(int)

        # Combine text
        self.df['description'] = (
            self.df['title'].fillna('') + ' ' + self.df['sub_title'].fillna('')
        )

        self.df['desc_length'] = self.df['description'].str.len()
        self.df['word_count'] = self.df['description'].str.split().str.len()

        # Text features
        self.df['more_than'] = self.df['description'].str.contains(
            '|'.join(PATTERNS_MORE), 
            regex=True
        ).astype(int)
        self.df['less_than'] = self.df['description'].str.contains(
            '|'.join(PATTERNS_LESS), 
            regex=True
        ).astype(int)
        
        # More robust player counting
        self.df['is_and'] = self.df['description'].str.contains(
            r'\b(?:et)\b', 
            regex=True
        ).astype(int)

        # don't count the 'ou plus' or 'ou moins' in the is_or feature
        self.df['is_or'] = self.df['description'].str.contains(
            r'\b(?:ou)\b(?!\s+(?:plus|moins))', 
            regex=True
        ).astype(int)

    def _embed_text(self, texts: pd.Series) -> np.ndarray:
        """Generate text embeddings with the right stopwords."""
        french_stopwords = get_stop_words('french')
        
        if self.text_embedding_method == 'tfidf':
            return TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 3),
                stop_words=french_stopwords
            ).fit_transform(texts).toarray()
        
        elif self.text_embedding_method == 'count':
            return CountVectorizer(
                max_features=1000,
                ngram_range=(1, 3),
                stop_words=french_stopwords
            ).fit_transform(texts).toarray()
        
        elif self.text_embedding_method in ('camembert', 'hybrid'):
            embeddings = []
            for i in tqdm(range(0, len(texts), self.batch_size_embedding), desc="Embedding CamemBERT"):
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

    def _build_preprocessor(self) -> ColumnTransformer:
        """Build the preprocessing pipeline."""
        # Numeric features
        numeric_features = [
            'old_odd', 'odd', 'odds_ratio', 'odds_diff', 'odds_percentage_change',
            'desc_length', 'word_count', 'more_than', 'less_than', 'is_and', 'is_or'
        ]
        if not self.without_golden:
            numeric_features.extend(['golden_odds_boost', 'is_golden'])
        
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical features
        categorical_transformer = Pipeline([
            ('target_enc', TargetEncoder()),
            ('scaler', StandardScaler())
        ])
        
        # Text features
        if self.use_text:
            text_transformer = Pipeline([
                ('clean', TextPreprocessor(lowercase=True, lemmatize=True)),
                ('vectorize', FunctionTransformer(self._embed_text, validate=False)),
                ('select', SelectFromModel(RandomForestClassifier(n_estimators=50)))
            ])

            if self.text_embedding_method in ('tfidf', 'count'):
                self.text_feature_names = None  # Will be set during fit
        
        transformers = [
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, ['sport']),
            ('text', text_transformer, 'description') 
        ] if self.use_text else [
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, ['sport'])
        ]

        return ColumnTransformer(transformers, remainder='drop')

    def _prepare_features(self, df: pd.DataFrame, y: pd.Series = None) -> torch.Tensor:
        """Convert DataFrame to tensor after preprocessing"""
        if df is None or df.empty:
            return None
            
        try:
            if not hasattr(self, 'preprocessor'):
                self.preprocessor = self._build_preprocessor()
                X = self.preprocessor.fit_transform(df, y)

                if self.use_text and self.text_embedding_method in ('tfidf', 'count'):
                    # Get feature names from the vectorizer
                    text_transformer = self.preprocessor.named_transformers_['text']
                    vectorizer = text_transformer.named_steps['vectorize']
                    if hasattr(vectorizer, 'get_feature_names_out'):
                        self.text_feature_names = vectorizer.get_feature_names_out()
                    
                    # Store which features were selected
                    selector = text_transformer.named_steps['select']
                    self.selected_text_features = selector.get_support()
                    
                    # Get the names of the selected features
                    if hasattr(self, 'text_feature_names'):
                        self.selected_text_feature_names = self.text_feature_names[self.selected_text_features]
            else:
                X = self.preprocessor.transform(df)
            
            return torch.FloatTensor(X) if X is not None else None
            
        except Exception as e:
            print("error: ", str(e))
            raise

    def log_parameters(self, mlflow: MLFlow) -> None:
        """Log parameters for MLFLOW"""
        mlflow.log_params({"text_embedding_method": self.text_embedding_method,
                           "use_class_token": self.use_class_token,
                           "test_size": self.test_size,
                           "val_size": self.val_size,
                           "chronological_split": self.chronological_split,
                           "random_state": self.random_state,
                           "batch_size_embedding": self.batch_size_embedding,
                           "threshold_sports": self.threshold_sports,
                           "only_golden": self.only_golden,
                           "without_golden": self.without_golden,
                           "with_feature_is_golden": self.with_feature_is_golden,
                           "use_text": self.use_text
                        })

    def __call__(self) -> None:
        """Run the complete data preparation."""
        self._filter_data()

        # Print 10 random rows of the dataframe
        print("Sample of 10 rows from the DataFrame:")
        sample = self.df.sample(10, random_state=self.random_state)
        for index, row in sample.iterrows():
            print(f"Row {index}:")
            print(f"  Title: {row['title']}")
            print(f"  Sub-title: {row['sub_title']}")
            print(f"  Description: {row['description']}")
            print(f"  Sport: {row['sport']}")
            print(f"  Result: {row['result']}")
            print(f"  Old Odd: {row['old_odd']}, New Odd: {row['odd']}")
            print(f"  Golden: {row['golden']}")
            print(f"  Odds Ratio: {row['odds_ratio']}")
            print(f"  less_than: {row['less_than']}, more_than: {row['more_than']}")
            print(f"  is_and: {row['is_and']}, is_or: {row['is_or']}")

        # Split the data
        if self.chronological_split:
            self.df = self.df.sort_values('ID')
            test_cutoff = int(len(self.df) * (1 - self.test_size))
            self.final_train_df, self.final_test_df = self.df.iloc[:test_cutoff], self.df.iloc[test_cutoff:]
            self.final_val_df = None
        else:
            shuffled_df = self.df.sample(frac=1, random_state=self.random_state)
            test_cutoff = int(len(shuffled_df) * (1 - self.test_size))
            train_val_df, self.final_test_df = shuffled_df.iloc[:test_cutoff], shuffled_df.iloc[test_cutoff:]
            
            if self.val_size:
                val_cutoff = int(len(train_val_df) * (1 - self.val_size))
                self.final_train_df, self.final_val_df = train_val_df.iloc[:val_cutoff], train_val_df.iloc[val_cutoff:]
            else:
                self.final_train_df, self.final_val_df = train_val_df, None

        print("exemple of final train df:")
        print(self.final_train_df.head())
        print("columns of final train df:")
        print(self.final_train_df.columns)

        # Preparation of targets
        y_train = self.final_train_df["result"].map({'Gagné': 1, 'Perdu': 0})
        y_test = self.final_test_df["result"].map({'Gagné': 1, 'Perdu': 0})
        y_val = self.final_val_df["result"].map({'Gagné': 1, 'Perdu': 0}) if self.final_val_df is not None else None

        # Preparation of features
        self.X_train = self._prepare_features(self.final_train_df, y_train)
        self.X_test = self._prepare_features(self.final_test_df)
        self.X_val = self._prepare_features(self.final_val_df) if self.final_val_df is not None else None

        # Conversion to tensors
        self.y_train = torch.LongTensor(y_train.values)
        self.y_test = torch.LongTensor(y_test.values)
        self.y_val = torch.LongTensor(y_val.values) if y_val is not None else None