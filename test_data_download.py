# -*- coding: utf-8 -*-
import sys
import io
# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import yfinance as yf
from datetime import datetime, timedelta

print("Testing data download...")

end_date = datetime.now()
start_date = end_date - timedelta(days=365)

ticker = 'NVDA'
print(f"Downloading {ticker} data from {start_date.date()} to {end_date.date()}...")

try:
    df = yf.download(ticker, start=start_date, end=end_date, progress=True)
    print(f"\nDownload successful!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nLast few rows:")
    print(df.tail())
except Exception as e:
    print(f"Download failed: {e}")
    import traceback
    traceback.print_exc()
