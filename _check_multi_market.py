import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
import yfinance as yf

# Test various ticker formats
tickers = {
    # TW
    '3131.TW':'弘塑', '3563.TW':'牧德', '5289.TW':'宜鼎',
    '6640.TW':'均華', '7734.TW':'印能科技', '7751.TW':'立騰',
    '7769.TW':'鴻勁', '7822.TW':'倍利科', '8299.TW':'群聯',
    # Korean
    '000660.KS':'海力士', '005930.KS':'三星',
    # China A-share
    '002371.SZ':'北方華創', '300604.SZ':'長川科技',
    '603986.SH':'兆易創新', '688037.SH':'芯源微',
    '688072.SH':'拓荊科技', '688082.SS':'盛美上海', '688383.SH':'新益昌',
    # Japan
    '285A.T':'鐵俠', '6857.T':'愛德萬', '8035.T':'東京威力',
    # US (already trained, just verify)
    'LRCX':'科林研發',
}

print(f'{"Ticker":<15} {"Rows":>5}  Company')
print('-'*45)
for t, name in tickers.items():
    try:
        d = yf.Ticker(t).history(period='5d')
        rows = len(d)
        status = '✅' if rows > 0 else '❌'
        print(f'{status} {t:<13} {rows:>5}  {name}')
    except Exception as e:
        print(f'❌ {t:<13}     0  {name} ({e})')
