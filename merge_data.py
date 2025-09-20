import pandas as pd
from pathlib import Path
from news_data import NewsData, build_sentiment
from stock_data import StockDownloads
from glob import glob
import os
import re

class Merger:
    def __init__(self, news_path='data/news', stock_path='data/stocks'):
        self.news_path = Path(news_path)
        self.stock_path = Path(stock_path)

    def load_news(self) -> pd.DataFrame:
        news_files = glob(str(self.news_path / 'news_*.csv'))
        if not news_files:
            print('No news files found')
            return pd.DataFrame()
        news_df = pd.concat([pd.read_csv(f) for f in news_files], ignore_index=True)
        if 'published_date' in news_df.columns:
            news_df.rename(columns={'published_date': 'date'}, inplace=True)

        if 'sentiment' not in news_df.columns:
            news_df['sentiment'] = 0.0

        news_df['date'] = pd.to_datetime(news_df['date']).dt.strftime('%Y-%m-%d')
        return news_df

    def load_stocks(self) -> pd.DataFrame:
        stock_files = [
            f for f in glob(str(self.stock_path / '*.csv'))
            if not any(x in f for x in ('news_', 'index_', 'merged'))
               and re.search(r'[a-z]+\_\d{4}\.csv$', os.path.basename(f), re.IGNORECASE)
        ]
        stock_dfs = []
        for f in stock_files:
            df = pd.read_csv(f)
            if 'symbol' not in df.columns:
                print(f"Skipping {f}, no 'symbol' column.")
                continue
            if 'Date' in df.columns:
                df.rename(columns={'Date': 'date'}, inplace=True)
            if 'date' not in df.columns:
                print(f"Skipping {f}, missing 'date' column.")
                continue
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            stock_dfs.append(df)
        if not stock_dfs:
            print('No valid stock files with symbol and date found.')
            return pd.DataFrame()

        return pd.concat(stock_dfs, ignore_index=True)

    def merge_data(self, save_path=None) -> pd.DataFrame:
        if save_path and Path(save_path).exists():
            print(f'Skipping merge. Already exists at: {save_path}')
            return pd.read_csv(save_path)
        news_df = self.load_news()
        if news_df.empty:
            return pd.DataFrame()

        stock_df = self.load_stocks()
        if stock_df.empty:
            return pd.DataFrame()

        sentiment_avg = news_df.groupby(['symbol', 'date'])['sentiment'].mean().reset_index()

        merged_df = pd.merge(stock_df, sentiment_avg, on=['symbol', 'date'], how='left')
        merged_df['sentiment'] = merged_df['sentiment'].fillna(0.0)

        merged_df['Open'] = pd.to_numeric(merged_df['Open'], errors='coerce')
        merged_df['Close'] = pd.to_numeric(merged_df['Close'], errors='coerce')

        merged_df['percent_change'] = ((merged_df['Close'] - merged_df['Open']) / merged_df['Open']) * 100
        merged_df['date'] = pd.to_datetime(merged_df['date'])
        merged_df.sort_values(by='date', ascending=False, inplace=True)
        if save_path:
            merged_df.to_csv('data/stock_news.csv', index=False)
            print('Final merged data saved to data/stock_news.csv')
        return merged_df