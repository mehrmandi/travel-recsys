from src.preprocessing.audiala_preprocessor import AudialaPreprocessor

def main():
    filepath = "data/raw/audiala_places.csv"
    
    preprocessor = AudialaPreprocessor(filepath)
    
    df = preprocessor.load_data()
    
    preprocessor.inspect_schema()

    df_clean = preprocessor.clean_data()

    print(f"Clean shape: {df_clean.shape}")

    df_mvp = preprocessor.filter_for_mvp(
        specific_cities=[
            "Paris",
            "Warsaw",
            "Prague",
            "London",
            "New York City",
        ]
    )
    
    train_df, val_df, test_df = preprocessor.create_splits(
        test_size=0.2,
        val_size=0.1,
        random_state=42
        )

    preprocessor.save_splits(train_df, val_df, test_df, output_dir="data/processed")


if __name__ == "__main__":
    main()
