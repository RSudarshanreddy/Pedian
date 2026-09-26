# Which EXIT keeps the upside without giving it back?
#
# The owner's current rule is roughly "sell when it is up a lot". Measured over
# 23 real sells that was right on direction 52% of the time, but the saves were
# 2-8% and the misses 20-29%, so the stock averaged +4.15% over the 15 sessions
# AFTER each sell. The objection behind the rule is real though -- names do give
# it back (BALUFORGE -19.9%). So the question is not hold-vs-sell, it is which
# trigger.
#
# Entry is fixed: top 3 per day by Expected_Move, at the next open. Only the
# exit varies. Every result is net of one round trip.
import warnings, sys, logging; warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
sys.path.insert(0, "C:/Users/Lenovo/Desktop/stocks/Pedian/scanner/src")
import pandas as pd, numpy as np, yfinance as yf, momentum as M

cfg = M.ScannerConfig(); COST = cfg.round_trip_cost_pct; MH = cfg.move_horizon_days
L = 60; MAXHOLD = 30; STEP = 2


def froll(a, w, fn):
    return getattr(pd.Series(a).rolling(w), fn)().shift(-(w - 1)).to_numpy()


def run_exit(entry, highs, lows, closes, atr0, rule, param):
    """Walk the trade forward day by day. Returns (pct, sessions_held)."""
    peak = entry
    for k in range(len(closes)):
        peak = max(peak, highs[k])
        if rule == "trail_pct":
            if lows[k] <= peak * (1 - param / 100):
                return (peak * (1 - param / 100) / entry - 1) * 100 - COST, k + 1
        elif rule == "trail_atr":
            stop = peak - param * atr0
            if lows[k] <= stop:
                return (stop / entry - 1) * 100 - COST, k + 1
        elif rule == "take_profit":
            if highs[k] >= entry * (1 + param / 100):
                return param - COST, k + 1
        elif rule == "owner":
            # approximate the observed behaviour: bank it once up `param`
            if closes[k] >= entry * (1 + param / 100):
                return (closes[k] / entry - 1) * 100 - COST, k + 1
    return (closes[-1] / entry - 1) * 100 - COST, len(closes)


tk = M.load_nse_tickers(); rows = []; CH = M.chunks(tk, 150)
for ci, chunk in enumerate(CH, 1):
    try:
        raw = yf.download(chunk, period="2y", interval="1d", progress=False,
                          auto_adjust=True, group_by="ticker", threads=True)
    except Exception:
        continue
    for t in chunk:
        try: f = M.get_ticker_frame(raw, t)
        except Exception: continue
        if f.empty or len(f) < 460: continue
        d = M.add_indicators(f, cfg)
        c = d["Close"].to_numpy(); o = d["Open"].to_numpy()
        hi = d["High"].to_numpy(); lo = d["Low"].to_numpy()
        atr = d["ATR"].to_numpy(); vm = d["volatility_measure"].to_numpy()
        acm = d["abs_close_move"].to_numpy(); tv = d["avg_traded_value20_cr"].to_numpy()
        ivd = d["is_volatile_day"].to_numpy(); pvm = d["past_vol_mean"].to_numpy()
        n = len(d)
        pk = (np.append(froll(c, MH, "max")[1:], np.nan) - c) / c * 100 - COST
        elig = pvm >= cfg.min_avg_volatility
        for i in range(300, n - MAXHOLD - 2, STEP):
            px = c[i]
            if not (cfg.min_price < px < cfg.max_price): continue
            if not np.isfinite(tv[i]) or tv[i] < cfg.min_avg_traded_value_cr: continue
            entry = o[i + 1]
            if not np.isfinite(entry) or entry <= 0 or not np.isfinite(atr[i]): continue
            end = i - MH; lb = slice(max(0, end - L), end)
            p = pk[lb][elig[lb]]; p = p[np.isfinite(p)]
            if len(p) < cfg.min_persistence_sample: continue
            typ = float(np.median(p))
            if typ < cfg.min_typical_move_pct: continue
            h = vm[max(0, i - L):i]; h = h[np.isfinite(h)]
            if len(h) < 40 or float(np.mean(h)) < cfg.min_avg_volatility: continue
            if float(np.median(h)) < cfg.min_median_volatility: continue
            if int(ivd[max(0, i - L):i].sum()) < cfg.min_volatile_days: continue
            mv_ = acm[max(0, i - 180):i]; mv_ = mv_[np.isfinite(mv_)]
            if len(mv_) < 120: continue
            if float(np.percentile(mv_, 95)) / max(float(np.median(mv_)), .01) > cfg.max_spike_ratio: continue
            w = c[max(0, i - 250):i + 1]; pa = np.maximum.accumulate(w)
            if float(((w - pa) / pa).min() * 100) < cfg.min_drawdown_pct: continue
            H = hi[i + 1:i + 1 + MAXHOLD]; LO = lo[i + 1:i + 1 + MAXHOLD]; C = c[i + 1:i + 1 + MAXHOLD]
            if len(C) < MAXHOLD or not np.all(np.isfinite(C)): continue
            rec = dict(sym=t, date=d.index[i], typ=typ)
            rec["hold30"], rec["hold30_d"] = (C[-1] / entry - 1) * 100 - COST, MAXHOLD
            for pct in (8, 12, 15, 20):
                rec[f"trail{pct}"], rec[f"trail{pct}_d"] = run_exit(entry, H, LO, C, atr[i], "trail_pct", pct)
            for m in (2.0, 3.0):
                k = str(m).replace(".", "")
                rec[f"atr{k}"], rec[f"atr{k}_d"] = run_exit(entry, H, LO, C, atr[i], "trail_atr", m)
            for tp in (10, 15):
                rec[f"owner{tp}"], rec[f"owner{tp}_d"] = run_exit(entry, H, LO, C, atr[i], "owner", tp)
            rows.append(rec)
    print(f"  chunk {ci}/{len(CH)}: {len(rows):,}", flush=True)
pd.DataFrame(rows).to_csv("exits.csv", index=False)
print(f"DONE {len(rows):,}")
