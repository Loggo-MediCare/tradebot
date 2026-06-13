import re

stocks = [
    ('mrna', 'MRNA', 'Moderna', '60.44'),
    ('omc', 'OMC', 'Omnicom', '88.79'),
    ('grmn', 'GRMN', 'Garmin', '84.34'),
    ('tpl', 'TPL', 'Texas Pacific Land', '71.71'),
    ('coin', 'COIN', 'Coinbase', '65.85'),
    ('gpn', 'GPN', 'Global Payments', '80.07'),
    ('moh', 'MOH', 'Molina Healthcare', '72.78'),
    ('bax', 'BAX', 'Baxter', '75.44'),
    ('cien', 'CIEN', 'Ciena', '62.99'),
]

for file_ticker, ticker, name, accuracy in stocks:
    filename = f'get_trading_signal_{file_ticker}.py'
    
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace all occurrences
    content = content.replace('AMAT (Applied Materials 應用材料)', f'{ticker} ({name})')
    content = content.replace('AMAT', ticker)
    content = content.replace('amat', file_ticker)
    content = content.replace('Applied Materials', name)
    content = content.replace('69.73', accuracy)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f'✅ 更新 {filename}')

print('\n所有檔案更新完成！')
