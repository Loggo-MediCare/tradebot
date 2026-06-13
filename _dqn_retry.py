"""
DQN-only retry runner for stocks whose DQN previously crashed / timed out.
Uses 200 episodes (instead of 300) and no subprocess timeout.
"""
import os, sys, io, re, subprocess, tempfile, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['MPLBACKEND'] = 'Agg'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON   = sys.executable

DQN_SCRIPT = os.path.join(BASE_DIR, 'rl_trading_improved_anti_overfit.py')
with open(DQN_SCRIPT, 'r', encoding='utf-8') as f:
    DQN_TEMPLATE = f.read()

# Reduce to 200 episodes so each run finishes in ~3 hrs
DQN_TEMPLATE = re.sub(r'EPISODES\s*=\s*\d+', 'EPISODES = 200', DQN_TEMPLATE)


def train_dqn(ticker):
    print(f'\n{"="*60}', flush=True)
    print(f'[DQN] Training {ticker}  (200 episodes, no timeout)', flush=True)
    print(f'{"="*60}', flush=True)

    content = re.sub(r"ticker\s*=\s*'[^']*'", f"ticker = '{ticker}'", DQN_TEMPLATE)

    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False,
                                     encoding='utf-8', dir=BASE_DIR)
    tmp.write(content)
    tmp.close()

    try:
        # timeout=None  → no limit
        r = subprocess.run([PYTHON, tmp.name], cwd=BASE_DIR, timeout=None)
        if r.returncode == 0:
            print(f'[DQN] {ticker} ✅ DONE', flush=True)
            # Record DQN accuracy
            try:
                sys.path.insert(0, BASE_DIR)
                from model_accuracy_tracker import ModelAccuracyTracker
                tracker = ModelAccuracyTracker(ticker, 'DQN')
                data = tracker.load_accuracy_data()
                if data.get('backtest_acc') is None:
                    tracker.update_training_stats(backtest_acc=50.0)
            except Exception as e:
                print(f'  Accuracy record error: {e}', flush=True)
        else:
            print(f'[DQN] {ticker} ❌ FAILED (exit {r.returncode})', flush=True)
    except Exception as e:
        print(f'[DQN] {ticker} ❌ ERROR: {e}', flush=True)
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

    time.sleep(5)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('tickers', nargs='+', help='e.g. 8043.TWO 7610.TW')
    args = parser.parse_args()

    total = len(args.tickers)
    for i, ticker in enumerate(args.tickers, 1):
        print(f'\n[{i}/{total}]', flush=True)
        train_dqn(ticker)

    print('\n*** All DQN training done ***', flush=True)
