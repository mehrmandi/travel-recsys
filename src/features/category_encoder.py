"""
Category Feature Encoding Module.

Responsibilities:
- Transform categorical features (e.g. 'category') into pure One-Hot representations.
- Persist category vocabulary/order for deterministic encoding across train/val/test splits.
"""

import pandas as pd
import numpy as np
from typing import List, Optional

class CategoryEncoder:
    def __init__(self, target_column: str = "category", unknown_category: str = "unknown"):
        self.target_column = target_column
        self.unknown_category = unknown_category.lower()
        self.categories: Optional[List[str]] = None
        
    def fit(self, df: pd.DataFrame) -> "CategoryEncoder":
        if self.target_column not in df.columns:
            raise ValueError(f"Column '{self.target_column}' does not exist in dataframe.")
        
        raw_categories = (df[self.target_column].dropna().astype(str).str.strip().str.lower().unique().tolist())
        
        categories_set = set(raw_categories)
        categories_set.add(self.unknown_category)
        
        self.categories = sorted(list(categories_set))
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.categories is None:
            raise RuntimeError(
                "categoryEncoder must be fitted before calling transform().")
            
        if self.target_column not in df.columns:
            raise ValueError(f"Column '{self.target_column}' does not exist in input dataframe.")
        
        series = (df[self.target_column].fillna(self.unknown_category).astype(str).str.strip().str.lower())
        
        series = series.replace("", self.unknown_category)
        
        known_set = set(self.categories)
        series = series.apply(lambda cat: cat if cat in known_set else self.unknown_category)
        
        cat_series = pd.Categorical(series, categories=self.categories)
        one_hot = pd.get_dummies(cat_series, prefix="cat", dtype=np.float32)
        
        return one_hot
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
    
    def get_feature_names(self) -> List[str]:
        if self.categories is None:
            return []
        
        return [f"cat_{cat}" for cat in self.categories]
           
    
