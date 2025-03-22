import yfinance as yf
from datetime import datetime, timedelta
import csv

# Obtain stock data from yfinance
ticker = "TSLA"
start_date = "2010-06-29"
end_date = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")

data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
data.to_csv("TSLA_Data.csv")

# Modify header to proper values
with open('TSLA_Data.csv', 'r', newline='') as infile:
    reader = list(csv.reader(infile))
    reader[0] = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    reader.pop(1)
    reader.pop(1)

with open('TSLA_Data.csv', 'w', newline='') as outfile:
    csv.writer(outfile).writerows(reader)