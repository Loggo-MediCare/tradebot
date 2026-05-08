from FinMind.data import DataLoader
import pandas as pd

dl = DataLoader()
# 不登入也可以使用（有請求次數限制）

# 取得日資料
df = dl.taiwan_stock_daily(
    stock_id='2330',
    start_date='2024-01-01',
    end_date='2025-01-26'
)
print(df)
print(f"\n共 {len(df)} 筆資料")