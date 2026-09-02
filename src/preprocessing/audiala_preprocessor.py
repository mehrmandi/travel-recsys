"""
Audiala Preprocessing Module.
data sanitization, MVP subset filtering, and dataset splitting.
"""


import pandas as pd
import numpy as np
from typing import Tuple, List, Optional

class AudialaPreprocessor:
    """
    Data preprocessor for the Audiala Places dataset.
    """
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        self.df = pd.read_csv(self.filepath)
        return self.df
        
    def inspect_schema(self) -> None:
        if self.df is None:
            raise ValueError("Dataset is not loaded. Call load_data() first.")
        
        print("--- Column Information and Dimensions ---")
        print(f"Dataset Shape: {self.df.shape}")
        print(self.df.info())
        print("\n--- Missing Value Ratio (%) Per Column ---")
        print((self.df.isnull().mean() * 100).round(2))
    
    def clean_data(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("Dataset is not loaded. Call load_data() first.")
        
        df_clean = self.df.copy()
        
        required_cols = ["name", "latitude", "longitude"]
        
        missing_cols = [
            col for col in required_cols
            if col not in df_clean.columns
        ]


        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}"
            )
            
        df_clean = df_clean.dropna(subset=required_cols)
        
        if "latitude" in df_clean.columns and "longitude" in df_clean.columns:
            df_clean["longitude"] = pd.to_numeric(df_clean["longitude"], errors="coerce")
            df_clean["latitude"] = pd.to_numeric(df_clean["latitude"], errors="coerce")
            df_clean = df_clean.dropna(subset=["latitude", "longitude"])
            
            valid_coords = (
                (df_clean["latitude"] >= -90.0) & (df_clean["latitude"] <= 90.0) &
                (df_clean["longitude"] >= -180.0) & (df_clean["longitude"] <= 180.0)
            )
            df_clean = df_clean[valid_coords]
            
        if "id" in df_clean.columns:
            df_clean = df_clean.drop_duplicates(subset=["id"])
        
        else:
            df_clean = df_clean.drop_duplicates(subset=["name", "latitude", "longitude"])
            
        self.df = df_clean.reset_index(drop=True)
        print(f"Valid records remaining after sanitization: {len(self.df)}")
        return self.df
    
    def filter_for_mvp(self, top_n_cities: int = 5, specific_cities: Optional[list[str]] = None) -> pd.DataFrame:
        if self.df is None or "city" not in self.df.columns:
            return self.df
        
        if specific_cities:
            return self.df[self.df["city"].isin(specific_cities)].reset_index(drop=True)
        
        top_cities = self.df["city"].value_counts().head(top_n_cities).index
        return self.df[self.df["city"].isin(top_cities)].reset_index(drop=True)
    
    def create_splits(self, test_size: float= 0.2, val_size: float= 0.1, random_state: int= 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
            

            
        
    
    