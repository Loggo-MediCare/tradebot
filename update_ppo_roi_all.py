"""
Bulk-update all trading signal scripts to use format_ppo_roi_line from ppo_backtest_cache.
Run once; idempotent (skips files already containing format_ppo_roi_line).
"""
import os, re, sys
sys.stdout.reconfigure(encoding='utf-8')

BASE = os.path.dirname(os.path.abspath(__file__))

TARGET_FILES = [
    'get_trading_signal_2881.py',
    'get_trading_signal_aaoi.py',
    'get_trading_signal_amat.py',
    'get_trading_signal_amd.py',
    'get_trading_signal_amzn.py',
    'get_trading_signal_ARM.py',
    'get_trading_signal_axon.py',
    'get_trading_signal_bax.py',
    'get_trading_signal_cien.py',
    'get_trading_signal_coin.py',
    'get_trading_signal_gild.py',
    'get_trading_signal_googl.py',
    'get_trading_signal_gpn.py',
    'get_trading_signal_grmn.py',
    'get_trading_signal_hsai.py',
    'get_trading_signal_intc.py',
    'get_trading_signal_invz.py',
    'get_trading_signal_ionq.py',
    'get_trading_signal_jazz.py',
    'get_trading_signal_meta.py',
    'get_trading_signal_moh.py',
    'get_trading_signal_mpwr.py',
    'get_trading_signal_mrna.py',
    'get_trading_signal_msft.py',
    'get_trading_signal_mu.py',
    'get_trading_signal_omc.py',
    'get_trading_signal_orcl.py',
    'get_trading_signal_pltr.py',
    'get_trading_signal_rdw.py',
    'get_trading_signal_SMCI.py',
    'get_trading_signal_tpl.py',
    'get_trading_signal_txn.py',
]


def extract_ticker(txt):
    """Extract ticker from TICKER constant or yf.download call."""
    m = re.search(r"^TICKER\s*=\s*['\"]([^'\"]+)['\"]", txt, re.MULTILINE)
    if m: return m.group(1)
    m = re.search(r"yf\.download\(['\"]([^'\"]+)['\"]", txt)
    if m: return m.group(1)
    return None


def extract_ppo_model(txt):
    """Extract PPO model path from PPO_FILE constant or PPO.load call."""
    m = re.search(r"^PPO_FILE\s*=\s*['\"]([^'\"]+)['\"]", txt, re.MULTILINE)
    if m: return m.group(1)
    m = re.search(r'PPO\.load\(["\']([^"\']+)["\']', txt)
    if m: return m.group(1)
    # mu-style: model_filename = "ppo_mu_improved"
    m = re.search(r'model_filename\s*=\s*["\']([^"\']+)["\']', txt)
    if m: return m.group(1)
    return None


updated = []
skipped = []
failed = []

for fname in TARGET_FILES:
    fpath = os.path.join(BASE, fname)
    if not os.path.exists(fpath):
        print(f'  MISSING: {fname}')
        failed.append(fname)
        continue

    with open(fpath, encoding='utf-8', errors='ignore') as f:
        txt = f.read()

    if 'format_ppo_roi_line' in txt:
        skipped.append(fname)
        continue

    ticker  = extract_ticker(txt)
    ppo_mdl = extract_ppo_model(txt)

    if not ticker or not ppo_mdl:
        print(f'  CANNOT EXTRACT: {fname}  ticker={ticker} ppo_mdl={ppo_mdl}')
        failed.append(fname)
        continue

    new_txt = txt

    # ── 2881.py: three separate print lines ───────────────────────────────────
    if '2881' in fname:
        old_block = (
            '        print(f"PPO 動作值: {ppo_action:+.4f}")\n'
            '        if ppo_action>0.1:\n'
            '            print(f"🟢 看多 (已持有→續抱 | 未持有→可進場)")\n'
            '        elif ppo_action<-0.1:\n'
            '            print(f"🔴 看空 (已持有→考慮減倉 | 未持有→勿追)")\n'
            '        else:\n'
            '            print(f"🟡 中性觀望")'
        )
        new_block = (
            '        from ppo_backtest_cache import format_ppo_roi_line\n'
            '        print(format_ppo_roi_line(CODE, TICKER, PPO_MODEL, df, ppo_action))'
        )
        if old_block in new_txt:
            new_txt = new_txt.replace(old_block, new_block)
        else:
            print(f'  BLOCK NOT FOUND: {fname}')
            failed.append(fname)
            continue

    # ── aaoi-style: ppo_signal_label ──────────────────────────────────────────
    elif 'ppo_signal_label' in txt:
        old_line = '        print(f"\\n🤖 PPO Action: {ppo_action:+.4f}  → {ppo_signal_label}")'
        new_line = (
            '        from ppo_backtest_cache import format_ppo_roi_line\n'
            f"        print(format_ppo_roi_line('{ticker}', '{ticker}', '{ppo_mdl}', df, ppo_action))"
        )
        if old_line in new_txt:
            new_txt = new_txt.replace(old_line, new_line)
        else:
            print(f'  LINE NOT FOUND (aaoi-style): {fname}')
            failed.append(fname)
            continue

    # ── mu/pltr-style: PPO_ACC ─────────────────────────────────────────────────
    elif 'PPO_ACC' in txt and 'ppo_action' in txt:
        # pltr variant (double space before →)
        old_pltr = '        print(f"  PPO     ({PPO_ACC}%): action={ppo_action:+.3f}        →  {ppo_signal}")'
        # mu variant (single space before →)
        old_mu   = '    print(f"  PPO     ({PPO_ACC}%): action={ppo_action:+.3f}  →  {ppo_signal}")'
        new_line_pltr = (
            '        from ppo_backtest_cache import format_ppo_roi_line\n'
            f"        print(format_ppo_roi_line('{ticker}', '{ticker}', '{ppo_mdl}', df, ppo_action))"
        )
        new_line_mu = (
            '    from ppo_backtest_cache import format_ppo_roi_line\n'
            f"    print(format_ppo_roi_line('{ticker}', '{ticker}', '{ppo_mdl}', df, ppo_action))"
        )
        if old_pltr in new_txt:
            new_txt = new_txt.replace(old_pltr, new_line_pltr)
        elif old_mu in new_txt:
            new_txt = new_txt.replace(old_mu, new_line_mu)
        else:
            print(f'  LINE NOT FOUND (pltr/mu-style): {fname}')
            failed.append(fname)
            continue

    # ── amd-style: double-brace {{ppo_action}} ─────────────────────────────────
    elif '{{ppo_action' in txt:
        old_line = '            print(f"   PPO Action: {{ppo_action:+.4f}}  → {{ppo_signal}}")'
        new_line = (
            '            from ppo_backtest_cache import format_ppo_roi_line\n'
            f"            print(format_ppo_roi_line('{ticker}', '{ticker}', '{ppo_mdl}', df, ppo_action))"
        )
        if old_line in new_txt:
            new_txt = new_txt.replace(old_line, new_line)
        else:
            print(f'  LINE NOT FOUND (amd-style): {fname}')
            failed.append(fname)
            continue

    # ── ARM-style: single-brace {ppo_action} ──────────────────────────────────
    elif '   PPO Action: {ppo_action' in txt:
        old_line = '            print(f"   PPO Action: {ppo_action:+.4f}  → {ppo_signal}")'
        new_line = (
            '            from ppo_backtest_cache import format_ppo_roi_line\n'
            f"            print(format_ppo_roi_line('{ticker}', '{ticker}', '{ppo_mdl}', df, ppo_action))"
        )
        if old_line in new_txt:
            new_txt = new_txt.replace(old_line, new_line)
        else:
            print(f'  LINE NOT FOUND (ARM-style): {fname}')
            failed.append(fname)
            continue

    else:
        print(f'  UNKNOWN PATTERN: {fname}')
        failed.append(fname)
        continue

    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(new_txt)
    print(f'  UPDATED: {fname}  ticker={ticker}  ppo={ppo_mdl}')
    updated.append(fname)

print(f'\n{"="*60}')
print(f'Updated: {len(updated)}  Skipped: {len(skipped)}  Failed: {len(failed)}')
if failed:
    print(f'Failed: {failed}')
