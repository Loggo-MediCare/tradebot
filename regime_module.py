# ==========================================
# Gamma Regime 判定（Deadband + Debounce）
# ==========================================
import json, os, time, math
from pathlib import Path

DEADBAND_M     = 200.0   # 靜態門檻（M USD）；可依流動性調整
CONFIRM_NEEDED = 2       # 連續幾次才確認切換
MAX_HISTORY    = 12      # 保留歷史筆數

def _state_path(symbol: str) -> Path:
    return Path(f'.gex_regime_{symbol.upper()}.json')

def _load_state(symbol: str) -> dict:
    p = _state_path(symbol)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {
        'history':          [],
        'confirmed_regime': 'NEUTRAL',
        'pending_regime':   None,
        'pending_count':    0,
    }

def _save_state(symbol: str, state: dict):
    try:
        _state_path(symbol).write_text(json.dumps(state, indent=2))
    except Exception:
        pass


def classify_regime_raw(net_gex: float, deadband: float) -> str:
    """單次原始分類，不做 debounce"""
    if   net_gex >  deadband: return 'POSITIVE'
    elif net_gex < -deadband: return 'NEGATIVE'
    else:                     return 'NEUTRAL'


def _calc_dynamic_deadband(history: list, static_db: float) -> float:
    """
    動態門檻 = 2 × std(近期 net_gex)
    上限：靜態門檻的 3 倍（防止大波動撐爆）
    下限：靜態門檻的 50%
    只用歷史中「已落在 [-10×static_db, +10×static_db]」的值算 std，
    排除離群值（如 +4000M 拉爆）。
    """
    if len(history) < 4:
        return static_db
    # 只用最近 4 筆（rolling window），排除離群值（> 5× static_db）
    recent = history[-4:]
    cap    = static_db * 5   # tight cap: 200×5=1000M for MU
    vals   = [h['net_gex'] for h in recent if abs(h['net_gex']) <= cap]
    if len(vals) < 3:
        return static_db
    mean  = sum(vals) / len(vals)
    std   = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
    dyn   = 2.0 * std
    # 上限：3× static，下限：0.5× static
    return max(min(dyn, static_db * 3), static_db * 0.5)


def get_confirmed_regime(
    symbol:         str,
    net_gex:        float,
    deadband:       float = DEADBAND_M,
    confirm_needed: int   = CONFIRM_NEEDED,
    max_history:    int   = MAX_HISTORY,
) -> dict:
    """
    完整 Deadband + Debounce 判定。

    Returns dict:
      confirmed_regime  : 'POSITIVE' / 'NEGATIVE' / 'NEUTRAL'
      raw_regime        : 本次原始分類
      net_gex           : 本次值
      deadband          : 靜態門檻
      dynamic_deadband  : 動態門檻（歷史 std）
      is_flip_zone      : abs(net_gex) < dynamic_deadband
      pending_regime    : 等待確認中的方向（或 None）
      pending_count     : 已累積幾次
      confirm_needed    : 需要幾次
      history_len       : 已有幾筆歷史
    """
    state         = _load_state(symbol)
    history       = state['history']
    confirmed     = state['confirmed_regime']
    pending       = state['pending_regime']
    pending_count = state['pending_count']

    # ── 本次原始分類（永遠用靜態門檻，確保一致可重現）───────────────────────
    raw = classify_regime_raw(net_gex, deadband)

    # ── 動態 deadband（只用歷史中 raw != POSITIVE 大離群值的部分）────────────
    # 只取最近 4 筆且 abs < 5×static_db 的值來算 std，避免 +4000M 拉爆
    dyn_db = _calc_dynamic_deadband(history, deadband)

    # ── Debounce ──────────────────────────────────────────────────────────
    if raw == confirmed:
        pending       = None
        pending_count = 0
    else:
        if raw == pending:
            pending_count += 1
        else:
            pending       = raw
            pending_count = 1

        if pending_count >= confirm_needed:
            confirmed     = pending
            pending       = None
            pending_count = 0

    # ── 更新歷史 ─────────────────────────────────────────────────────────
    history.append({
        'ts':        time.strftime('%Y-%m-%d %H:%M:%S'),
        'net_gex':   round(net_gex, 1),
        'raw':       raw,
        'confirmed': confirmed,
    })
    history = history[-max_history:]

    _save_state(symbol, {
        'history':          history,
        'confirmed_regime': confirmed,
        'pending_regime':   pending,
        'pending_count':    pending_count,
    })

    # is_flip_zone 用動態門檻（更貼近當前市況）
    is_flip_zone = abs(net_gex) < dyn_db

    return {
        'confirmed_regime': confirmed,
        'raw_regime':       raw,
        'net_gex':          net_gex,
        'deadband':         deadband,
        'dynamic_deadband': round(dyn_db, 1),
        'is_flip_zone':     is_flip_zone,
        'pending_regime':   pending,
        'pending_count':    pending_count,
        'confirm_needed':   confirm_needed,
        'history_len':      len(history),
    }


def regime_label(r: dict) -> str:
    cr = r['confirmed_regime']
    fz = r['is_flip_zone']
    ng = r['net_gex']
    db = r['dynamic_deadband']
    if fz:
        return f"FLIP ZONE（淨GEX {ng:+.0f}M，門檻±{db:.0f}M）"
    elif cr == 'POSITIVE':
        return "正 Gamma（穩定）"
    elif cr == 'NEGATIVE':
        return "負 Gamma（追漲追跌）"
    else:
        return f"NEUTRAL（淨GEX {ng:+.0f}M）"


def regime_is_negative(r: dict) -> bool:
    return r['confirmed_regime'] == 'NEGATIVE' and not r['is_flip_zone']

def regime_is_positive(r: dict) -> bool:
    return r['confirmed_regime'] == 'POSITIVE' and not r['is_flip_zone']

