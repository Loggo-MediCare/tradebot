import yfinance as yf, warnings, sys
warnings.filterwarnings('ignore')
for t in ['3081.TW','3081.TWO','6147.TW','6147.TWO','6612.TW','6612.TWO']:
    try:
        d = yf.Ticker(t).history(period='5d')
        name = yf.Ticker(t).info.get('shortName','') if len(d)>0 else ''
        print(f"{t}: {len(d)} rows {name}")
    except Exception as e:
        print(f"{t}: ERROR {e}")
