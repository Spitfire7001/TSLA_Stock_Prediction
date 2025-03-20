import yfinance as yf
from datetime import datetime, timedelta

ticker = "TSLA"
start_date = "2010-06-29"
end_date = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")  # Tomorrow's date

data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
data.to_csv("TSLA_Data.csv")
