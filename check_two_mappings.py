import importlib.util
import os

SCRIPT = os.path.join(os.path.dirname(__file__), 'run_all_local_tw_to_excel.py')
spec = importlib.util.spec_from_file_location('run_mod', SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

print('Code\tTicker\tName')
for s in getattr(mod, 'SIGNAL_SCRIPTS', []):
    code = s['name'].split()[0]
    ticker = mod.get_ticker_from_code(code)
    print(f"{code}\t{ticker}\t{s['name']}")
