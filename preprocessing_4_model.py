import pandas as pd
import numpy as np
from glob import glob
from pathlib import Path

def get_csv_files(folder: str | Path, pattern: str = '*.csv') -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError("Folder does not exist.")
    return sorted(folder.glob(pattern))

def load_csv_data(inputs: str | Path) -> pd.DataFrame:
    inputs = Path(inputs)

    if not inputs.exists():
        raise FileNotFoundError(f"No such file or directory: {inputs}")

    if inputs.is_file():
        return pd.read_csv(inputs)

    if inputs.is_dir():
        csv_files = sorted(inputs.glob('*.csv'))
        if not csv_files:
            raise ValueError(f'No CSV files found in {inputs}')
        return pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    raise ValueError(f'Input must be a file of CSVs.')

def label_csv(inputs: str | Path) -> pd.DataFrame:
    df = pd.read_csv(inputs)
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
    if 'Volume' not in df.columns:
        df['Volume'] = 0
    df['Volume'] = pd.to_numeric(df.get('Volume', 0), errors='coerce').fillna(0.0)
    if 'sentiment' not in df.columns:
        df['sentiment'] = 0
    df['sentiment'] = pd.to_numeric(df.get('sentiment', 0), errors='coerce').fillna(0.0)
    if 'percent_change' not in df.columns:
        df['percent_change'] = (df['Close'] - df['Open']) / df['Open'] * 100

    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(['symbol', 'date'], inplace=True)
    df.drop_duplicates(subset=['symbol', 'date'], keep='last', inplace=True)
    df['Tomorrow'] = df.groupby('symbol')['Close'].shift(-1)
    df['Target'] = (df['Tomorrow'] > df['Close']).astype(int)
    df.dropna(subset=['Tomorrow'], inplace=True)
    df.to_csv('data/stock_news_labeled.csv', index=False)
    return df

def split(df, feature_cols=('sentiment','percent_change','Volume'), cutoff=None):

    df['date'] = pd.to_datetime(df['date'])
    if cutoff is None:
        cutoff = df['date'].max() - pd.Timedelta(days=180)
    elif isinstance(cutoff,str):
        cutoff = pd.Timestamp(cutoff)

    df = df.dropna(subset=list(feature_cols) + ['Target'])
    train = df[df['date'] < cutoff]
    test = df[df['date'] >= cutoff]

    if train.empty or test.empty:
        print(f'[split] Warning: train={len(train)} test={len(test)}. Consider a different cutoff or more data.')

    x_train = train[list(feature_cols)]
    y_train = train['Target']

    x_test = test[list(feature_cols)]
    y_test = test['Target']

    return x_train, x_test, y_train, y_test


