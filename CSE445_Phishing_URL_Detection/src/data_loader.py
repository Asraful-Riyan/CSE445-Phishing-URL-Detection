import pandas as pd
from sklearn.model_selection import train_test_split
from config import DATA_FILE, TARGET_COLUMN, URL_COLUMN, FEATURE_COLUMNS, RANDOM_STATE, TEST_SIZE


def load_dataset(path=DATA_FILE):
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Download LegitPhish and save it as data/url_features_extracted1.csv"
        )

    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    if TARGET_COLUMN not in df.columns:
        possible = [c for c in df.columns if c.lower() in {"classlabel", "label", "class", "target"}]
        if possible:
            df = df.rename(columns={possible[0]: TARGET_COLUMN})
        else:
            raise ValueError(f"Target column '{TARGET_COLUMN}' was not found. Columns: {list(df.columns)}")

    # Remove exact duplicate rows.
    df = df.drop_duplicates().reset_index(drop=True)

    # Convert target to numeric and keep valid binary labels.
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df = df[df[TARGET_COLUMN].isin([0, 1])].copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    if not available_features:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        available_features = [c for c in numeric_cols if c != TARGET_COLUMN]

    if len(available_features) < 2:
        raise ValueError("Not enough numeric feature columns were found in the dataset.")

    # Convert all selected features to numbers. Missing values are handled by pipelines.
    for col in available_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, available_features


def split_dataset(df, feature_columns):
    X = df[feature_columns].copy()
    y = df[TARGET_COLUMN].copy()

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


if __name__ == "__main__":
    df, features = load_dataset()
    print("Shape:", df.shape)
    print("Features:", features)
    print("\nClass distribution:")
    print(df[TARGET_COLUMN].value_counts().sort_index())
