import pandas as pd
import yfinance as yf
import time
from pathlib import Path

class StockDownloads:
    def __init__(self, indexes: dict, symbols: list, start_yr: int, end_yr: int, save: bool = True):
        self.indexes = indexes
        self.symbols = symbols
        self.start_yr = start_yr
        self.end_yr = end_yr
        self.save = save
        Path('data/stocks').mkdir(parents=True, exist_ok=True)
        Path('data/indexes').mkdir(parents=True, exist_ok=True)

    def get_index_data(self):
        #Create a folder called 'data' if it doesn't already exist

        index_dfs = []
        for name, symbol in self.indexes.items():
            for year in range(self.start_yr, self.end_yr + 1):
                file_path = f'data/indexes/index_{name.lower().replace(' ', '_')}_{year}.csv'
                if self.save and Path(file_path).exists():
                    print(f'Skipping index {name} for {year} (already exists)')
                    continue

                start_date = f'{year}-01-01'
                end_date = f'{year}-12-31'
                try:
                    print(f'Downloading index: {name} ({symbol} from {start_date} to {end_date})')
                    df = yf.download(symbol, start=start_date, end=end_date)
                    if df.empty:
                        print(f'No data for index {name} in {year}')
                        continue

                    df.reset_index(inplace=True)
                    df.insert(0, 'symbol', name.upper())
                    df.rename(columns={'Date':'date'},inplace=True)
                    if self.save:
                        df.to_csv(file_path, index=False)
                        print(f'Saved index: {name} -> {file_path}')
                    index_dfs.append(df)
                    time.sleep(5)

                except Exception as e:
                    print(f'Failed to download index {name}/{symbol} for {year}: {e}')
        return pd.concat(index_dfs, ignore_index=True) if index_dfs else pd.DataFrame()

    def get_stock_data(self):

        stock_dfs = []
        for symbol in self.symbols:
            for year in range(self.start_yr, self.end_yr + 1):
                file_path = f'data/stocks/{symbol.lower()}_{year}.csv'
                if self.save and Path(file_path).exists():
                    print(f'Skipping {symbol} {year} (already exists)')
                    continue

                start_date = f'{year}-01-01'
                end_date = f'{year}-12-31'
                try:
                    print(f'Downloading stock: {symbol} from {start_date} to {end_date}')
                    df = yf.download(symbol, start=start_date, end=end_date)
                    if df.empty:
                        print(f'No data for stock {symbol} in {year}')
                        continue

                    df.reset_index(inplace=True)
                    df.insert(0, 'symbol', symbol.upper())
                    df.rename(columns={'Date': 'date'}, inplace=True)
                    if self.save:
                        df.to_csv(file_path, index=False)
                        print(f'Saved stock: {symbol} -> {file_path}')
                    stock_dfs.append(df)
                    time.sleep(5)

                except Exception as e:
                    print(f'Failed to download stock {symbol} for {year}: {e}')
        return pd.concat(stock_dfs, ignore_index=True) if stock_dfs else pd.DataFrame()

    def run_stock_and_index_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self.get_index_data(), self.get_stock_data()