from src.cleaning import get_clean_data

if __name__ == "__main__":
    SEED = 69

    df = get_clean_data()
    print(df)