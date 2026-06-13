import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
import yfinance as yf

codes = ['3363','8291','4933','3357','5328','8044','8455']
for c in codes:
    for s in ['TW','TWO']:
        t = f"{c}.{s}"
        try:
            d = yf.Ticker(t).history(period='5d')
            if len(d) > 0:
                name = yf.Ticker(t).info.get('shortName','')
                print(f"FOUND: {t}  {name}")
                break
        except:
            pass
    else:
        print(f"NOT FOUND: {c}")
