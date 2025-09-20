from news_data import NewsData, build_sentiment
from stock_data import StockDownloads
from merge_data import Merger
from preprocessing_4_model import load_csv_data, label_csv, split
import pandas as pd

def main():
    symbols = [
        'NVDA', 'MSFT', 'AAPL', 'AMZN', 'META', 'AVGO', 'TSLA', 'BRK.B', 'GOOGL', 'GOOG', 'WMT',
        'AXP', 'AMGN', 'BA', 'CAT', 'CSCO', 'CVX', 'GS', 'HD', 'HON', 'IBM', 'JNJ',
        'RDDT', 'DUOL', 'SFM', 'MSTR', 'COKE', 'SMCI', 'CVNA', 'FIX', 'ALAB', 'ULS', 'ENSG'
    ]
    indexes = {
        'S&P500': '^GSPC',
        'NASDAQ': '^IXIC',
        'DOWJONES': '^DJI',
        'RUSSELL2000': '^RUT'
    }
    api_key = 'REMOVED'
    from_year = 2020
    to_year = 2025

    stock_downloader = StockDownloads(indexes=indexes, symbols=symbols, start_yr=from_year, end_yr=to_year)
    index_df, stock_df = stock_downloader.run_stock_and_index_data()

    news_getter = NewsData(api_key=api_key, from_year=from_year, to_year=to_year)
    news_df = news_getter.run(symbols)
    news_df = build_sentiment(news_df)
    if not news_df.empty:
        for symbol in symbols:
            symbol_df = news_df[news_df['symbol'] == symbol]
            if not symbol_df.empty:
                symbol_df.to_csv(f'data/news/news_{symbol.lower()}.csv', index=False)
        print(f'News data with sentiment saved to data/news/*.csv')
    else:
        print(f'No news data collected.')

    merger = Merger()
    merged_df = merger.merge_data(save_path='data/stock_news.csv')
    labeled_df = label_csv('data/stock_news.csv')
    x_train, x_test, y_train, y_test = split(labeled_df, feature_cols=('sentiment', 'percent_change', 'Volume'),
                                             cutoff=None)
    stock_df = load_csv_data('data/stocks')
    news_df = load_csv_data('data/news')
    indexes_df = load_csv_data('data/indexes')

if __name__ == "__main__":
    main()