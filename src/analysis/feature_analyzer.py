"""
Feature Analysis Module for Audiala Places Dataset.

Responsibilities:
- Statistical analysis of numerical features (PageRank, sitelinks, coordinates)
- Distribution and cardinality analysis for categorical features
- Correlation analysis between popularity signals
- Article tier vs. popularity analysis
- Geospatial distribution inspection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any

class FeatureAnalyzer:
    def __init__(self, df: pd.DataFrame):
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        
        self.df = df.copy()
        
    def analyze_categorical_features(self, columns: List[str] = None) -> Dict[str, pd.DataFrame]:
        if columns is None:
            columns = ["category", "article_tier", "city_en", "country_en"]
            
        results = {}
        
        print("\n" + "=" * 55)
        print("CATEGORICAL FEATURES ANALYSIS")
        print("=" * 55)
        
        for col in columns:
            if col not in self.df.columns:
                continue
            
            unique_count = self.df[col].nunique(dropna=True)
            value_counts = self.df[col].value_counts(dropna=False)
            top_counts = value_counts.head(10)
            top_percentages = (self.df[col].value_counts(normalize=True, dropna=False).head(10) * 100).round(2)
            
            summary = pd.DataFrame(
                {
                    "Count": top_counts,
                    "Percentage (%)": top_percentages
                }
            )
            
            results[col] = summary
            print(f"\nFeature: '{col}' | Unique Values: {unique_count}")
            print(summary)
            print("-" * 55)
        
        return results
    
    
    def analyze_numerical_features(self, columns: List[str] = None) -> pd.DataFrame:
        if columns is None:
            columns = ["wikidata_pagerank", "sitelinks", "latitude", "longitude"]
            
        valid_cols = [col for col in columns if col in self.df.columns]
        num_df = self.df[valid_cols].apply(pd.to_numeric, errors="coerce")
        
        stats = num_df.describe().T
        stats["skewness"] = num_df.skew().round(3)
        stats["missing_count"] = num_df.isna().sum()
        stats["missing_pct"] = (num_df.isna().mean() * 100).round(2)
        stats["zeros_count"] = (num_df == 0).sum()
        
        print("\n" + "=" * 55)
        print("NUMERICAL FEATURES STATISTICAL SUMMARY")
        print("=" * 55)
        print(stats)
        
        return stats
    
    
    def analyze_popularity_correlations(self) -> pd.DataFrame:
        signals = ["wikidata_pagerank", "sitelinks"]
        valid_cols = [col for col in signals if col in self.df.columns]
        
        if len(valid_cols) < 2:
            print("\nInsufficient separate popularity columns found (need >= 2).")
            return pd.DataFrame()
        
        corr_matrix = self.df[valid_cols].corr(method="spearman").round(4)
        
        print("\n" + "=" * 55)
        print("POPULARITY SIGNALS CORRELATION (Spearman)")
        print("=" * 55)
        print(corr_matrix)
        
        return corr_matrix
    
    
    def analyze_tier_vs_popularity(self) -> pd.DataFrame:
        if "article_tier" not in self.df.columns:
            return pd.DataFrame
        
        agg_dict = {}
        
        if "wikidata_pagerank" in self.df.columns:
            agg_dict["wikidata_pagerank"] = ["count", "median", "mean"]
        
        if "sitelinks" in self.df.columns:
            agg_dict["sitelinks"] = ["median", "mean"]
            
        grouped = self.df.groupby("article_tier").agg(agg_dict).round(3)
        
        print("\n" + "=" * 55)
        print("ARTICLE TIER VS POPULARITY SIGNALS")
        print("=" * 55)
        print(grouped)

        return grouped
    
    
    def analyze_geo_distribution(self, top_cities: int = 10) -> pd.DataFrame:
        if "city_en" not in self.df.columns:
            return pd.DataFrame
        
        city_counts = self.df["city_en"].value_counts().head(top_cities).index.tolist()
        top_df = self.df[self.df["city_en"].isin(city_counts)]
        
        geo_summary = top_df.groupby("city_en")[["latitude", "longitude"]].agg([
            "count", "mean", "std"]).round(3)
        
        print("\n" + "=" * 55)
        print(f"GEO DISTRIBUTION - TOP {top_cities} CITIES")
        print("=" * 55)
        print(geo_summary)

        return geo_summary
    
    
    def run_full_analysis(self) -> Dict[str, Any]:
        cat_results = self.analyze_categorical_features()
        num_results = self.analyze_numerical_features()
        corr_results = self.analyze_popularity_correlations()
        tier_results = self.analyze_tier_vs_popularity()
        geo_results = self.analyze_geo_distribution()
    
        return {
            "categorical": cat_results,
            "numerical": num_results,
            "correlation": corr_results,
            "tier_vs_popularity": tier_results,
            "geo_distribution": geo_results,
        }
            
    
            
        
    
    
        
    
