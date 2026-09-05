"""
Script to test and run feature analysis on the cleaned Audiala dataset.
"""

from src.analysis.feature_analyzer import FeatureAnalyzer
from src.preprocessing.audiala_preprocessor import AudialaPreprocessor
import sys
from pathlib import Path

# Ensure project root is accessible in Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))


def main():
    filepath = "data/raw/audiala_places.csv"

    # 1. Load and clean data using preprocessor
    preprocessor = AudialaPreprocessor(filepath)
    preprocessor.load_data()
    df_clean = preprocessor.clean_data()

    # 2. Initialize and run the full feature analysis
    analyzer = FeatureAnalyzer(df_clean)
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
