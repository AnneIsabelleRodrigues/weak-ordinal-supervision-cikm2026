import pandas as pd

def preprocess(
        df: pd.DataFrame,
        n_bins: int = 3,
        missing_strategy: str = "mode",
) -> pd.DataFrame:

    df = df.copy()

    if missing_strategy == "drop":
        df = df.dropna()
    else:
        for col in df.columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode().iloc[0])

    result = pd.DataFrame(index=df.index)

    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series) and series.nunique() > n_bins:
            result[col] = pd.cut(
                series,
                bins=n_bins,
                labels=False,
                duplicates="drop",
            ).astype(int)
        else:
            result[col] = pd.Categorical(series.astype(str)).codes.astype(int)

    result = result.astype(int)
    return result

