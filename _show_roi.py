import sys, io, os, json, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

base  = os.path.dirname(os.path.abspath(__file__))
files = glob.glob(os.path.join(base, 'model_accuracy_*.json'))

results = {}
for f in sorted(files):
    name  = os.path.basename(f).replace('model_accuracy_','').replace('.json','')
    parts = name.rsplit('_', 1)
    symbol = parts[0].replace('_','.') if len(parts)==2 else name.replace('_','.')
    mtype  = parts[1] if len(parts)==2 else 'PPO'

    with open(f, encoding='utf-8') as fh:
        d = json.load(fh)

    ba = d.get('backtest_accuracy')
    wr = d.get('win_rate')
    sr = d.get('sharpe_ratio')

    # Estimate ROI: new log formula -> invert; old formula -> (ba-50)*2
    if ba is not None and ba <= 100:
        roi_est = (ba - 50) * 2
    else:
        roi_est = None

    results[f'{symbol} [{mtype}]'] = {
        'ba': ba, 'roi': roi_est, 'wr': wr, 'sr': sr
    }

print(f'\n{"Stock [Model]":<32} {"Est. ROI":>10} {"Backtest Acc":>13} {"Win Rate":>9} {"Sharpe":>8}')
print('=' * 76)
for k, v in sorted(results.items(), key=lambda x: x[1]['roi'] or -9999, reverse=True):
    roi = f"{v['roi']:+.1f}%" if v['roi'] is not None else '  -'
    ba  = f"{v['ba']:.1f}"   if v['ba']  is not None else '  -'
    wr  = f"{v['wr']:.1f}%"  if v['wr']  is not None else '  -'
    sr  = f"{v['sr']:.3f}"   if v['sr']  is not None else '  -'
    print(f'{k:<32} {roi:>10} {ba:>13} {wr:>9} {sr:>8}')

print(f'\nTotal models tracked: {len(results)}')
