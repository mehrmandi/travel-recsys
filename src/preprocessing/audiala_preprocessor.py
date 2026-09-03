""" 
Audiala Preprocessing Module. 

Responsibilities:
- Load the Audiala Places dataset
- Validate and sanitize raw data
- Filter a small MVP subset
- Create reproducible train/validation/test splits 

"""


import pandas as pd
from typing import Tuple, Optional

class AudialaPreprocessor:
    """
    Data preprocessor for the Audiala Places dataset.
    """
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        self.df = pd.read_csv(self.filepath)
        print(f"Raw dataset shape: {self.df.shape}")
        return self.df
        
    def inspect_schema(self) -> None:
        if self.df is None:
            raise ValueError("Dataset is not loaded. Call load_data() first.")
        
        print("--- Column Information and Dimensions ---")
        print(f"Dataset Shape: {self.df.shape}")
        self.df.info()
        print("\n--- Missing Value Ratio (%) Per Column ---")
        print((self.df.isnull().mean() * 100).round(2))
    
    def clean_data(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("Dataset is not loaded. Call load_data() first.")
        
        df_clean = self.df.copy()
        
        required_cols = [
            "wikidata_id",
            "name_en",
            "latitude",
            "longitude",
            "country_iso2",
            "country_en",
            "city_en",
            "category",
        ]
        
        missing_cols = [
            col for col in required_cols
            if col not in df_clean.columns
        ]


        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}"
            )
            
        df_clean = df_clean.dropna(subset=required_cols)
        
        
        df_clean["longitude"] = pd.to_numeric(df_clean["longitude"], errors="coerce")
        df_clean["latitude"] = pd.to_numeric(df_clean["latitude"], errors="coerce")
        df_clean = df_clean.dropna(subset=["latitude", "longitude"])
            
        valid_coords = (
            df_clean["latitude"].between(-90, 90)
            & df_clean["longitude"].between(-180, 180)
        )
        
        df_clean = df_clean[valid_coords]
        
        string_columns = [
            "wikidata_id",
            "name_en",
            "country_iso2",
            "country_en",
            "city_en",
            "category"
            ]
            
        for col in string_columns:
            df_clean[col] = (df_clean[col].astype(str).str.strip())
            
        text_required = [
            "wikidata_id", 
            "name_en", 
            "country_en", 
            "city_en", 
            "category"
            ]
        
        non_empty_mask = (df_clean[text_required].ne("").all(axis=1))
        
        df_clean = df_clean.loc[non_empty_mask]
        
        df_clean = df_clean.drop_duplicates(subset=["wikidata_id"], keep='first')
            
        self.df = df_clean.reset_index(drop=True)
        
        print(f"Valid records remaining after sanitization: {len(self.df)}")
        return self.df
    
    
    def filter_for_mvp(self, top_n_cities: int = 5, specific_cities: Optional[list[str]] = None) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("Dataset is not loaded. Call load_data() first.")
        
        if "city_en" not in self.df.columns:
            raise ValueError("Column 'city_en' is missing from the dataset.")
        
        if top_n_cities <= 0:
            raise ValueError("top_n_cities must be greater than 0.")

        
        if specific_cities:
            if len(specific_cities) == 0:
                raise ValueError("specific_cities cannot be empty.")
            cities = set(specific_cities)
            
        else:  
            cities = set(self.df["city_en"].value_counts().head(top_n_cities).index)
            
        mvp_df = (self.df[self.df["city_en"].isin(cities)].reset_index(drop=True))
        
        print("\n--- MVP Dataset ---") 
        print(f"MVP shape: {mvp_df.shape}") 
        print("\nTop/Selected cities:") 
        print(mvp_df["city_en"] .value_counts())
        
        return mvp_df
    
    def create_splits(self, df: Optional[pd.DataFrame] = None, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self.df is None:
            raise ValueError("Dataset is not loaded. Call load_data() and clean_data() first.")
        
        if test_size + val_size >= 1.0:
            raise ValueError("Sum of test_size and val_size must be less than 1.0.")
        
        df_shuffled = self.df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        n_total = len(df_shuffled)
        n_test = int(n_total * test_size)
        n_val = int(n_total * val_size)
        
        test_df = df_shuffled.iloc[:n_test].reset_index(drop=True)
        val_df = df_shuffled.iloc[n_test:n_test + n_val].reset_index(drop=True)
        train_df = df_shuffled.iloc[n_test + n_val:].reset_index(drop=True)
        
        return train_df, val_df, test_df
            

            
        
    
    