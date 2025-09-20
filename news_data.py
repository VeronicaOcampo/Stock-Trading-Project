import pandas as pd
from pathlib import Path
import requests
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class NewsData:
    def __init__(self, api_key: str, from_year: int, to_year: int, save: bool=True):
        self.api_key = api_key.strip()
        self.from_year = from_year
        self.to_year = to_year
        self.save = save
        Path('data/news').mkdir(parents=True, exist_ok=True)
    def get_news_data(self, symbol: str, from_date: str, to_date: str):
        news_url = 'https://api.marketaux.com/v1/news/all'
        parameters = {
            'symbols': symbol,
            'api_token': self.api_key,
            'published_after': from_date,
            'published_before': to_date,
            'language': 'en'
        }
        response = requests.get(news_url, params=parameters)
        if response.status_code == 200:
            return response.json().get('data', [])
        if response.status_code == 429:
            print("Too many requests! You've hit your rate limit.")
        else:
            print(f'Error getting news for {symbol}: {response.status_code}')
        return []

    def convert_to_dataframe(self, symbol: str) -> pd.DataFrame:
        symbol_articles = []
        for year in range(self.from_year, self.to_year + 1, 2):
            start_year = year
            end_year = min(year + 1, self.to_year)
            path = f'data/news/news_{symbol.lower()}_{start_year}_{end_year}.csv'
            if self.save and Path(path).exists():
                print(f'Skipping {symbol} from {start_year} to {end_year} (already saved).')
                continue
            start_date = f'{start_year}-01-01'
            end_date = f'{end_year}-12-31'
            print(f'Getting news for {symbol} from {start_date} to {end_date}')
            try:
                articles = self.get_news_data(symbol, start_date, end_date)
                chunk_df = pd.DataFrame([{
                    'symbol': symbol,
                    'title': article.get('title'),
                    'published_date': article.get('published_at'),
                } for article in articles])
                if chunk_df.empty:
                    print(f'No articles for {symbol} from {start_date} to {end_date}')
                else:
                    if self.save:
                        chunk_df.to_csv(path, index=False)
                    symbol_articles.append(chunk_df)
            except Exception as e:
                print(f'Error downloading news for {symbol} from {start_year} to {end_year}: {e}')
            time.sleep(1)
        return pd.concat(symbol_articles, ignore_index=True) if symbol_articles else pd.DataFrame()
    def run(self, symbol_list: list) -> pd.DataFrame:
        total_articles = []
        for symbol in symbol_list:
            symbol_df = self.convert_to_dataframe(symbol)
            if not symbol_df.empty:
                total_articles.append(symbol_df)
        return pd.concat(total_articles, ignore_index=True) if total_articles else pd.DataFrame()
def build_sentiment(news_df: pd.DataFrame):
    analyzer = SentimentIntensityAnalyzer()
    for i in range(len(news_df)):
        sentiment = analyzer.polarity_scores(news_df.iloc[i]['title'])['compound']
        news_df.at[i, 'sentiment'] = sentiment
    return news_df