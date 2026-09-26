# The trailing stop failed because it follows price UP and then clips winners
# (big wins +45.76 -> +7.51). A FIXED stop, measured from the entry price, can
# only ever touch a position that is down -- it never moves, so a winner that
# has run 30% is never at risk of it. That is the asymmetric version: keep the
# 11.8pp saved on the big losers, pay nothing on the right tail.
#
# Also tested: the scanner's own Stop_Loss (entry - 1.5 * ATR), which every row
# already carries and which nobody has ever used.
import warnings, sys, logging; warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
sys.path.insert(0, "C:/Users/Lenovo/Desktop/stocks/Pedian/scanner/src")
import pandas as pd, numpy as np, yfinance as yf, momentum as M

cfg = M.ScannerConfig(); COST = cfg.round_trip_cost_pct; MH = cfg.move_horizon_days
L = 60; MAXHOLD = 30; STEP = 2


def froll(a, w, fn):
    return getattr(pd.Series(a).rolling(w), fn)().shift(-(w - 1)).to_numpy()


def fixed_stop(entry, lows, closes, stop_px):
    """Stop never moves. Returns (pct, sessions_held)."""
    for k in range(len(closes)):
        if lows[k] <= stop_px:
            return (stop_px / entry - 1) * 100 - COST, k + 1
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
        c = d["Close"].to_numpy(); o = d["Open"].to_numpy(); lo = d["Low"].to_numpy()
        atr = d["ATR"].to_numpy(); vm = d["volatility_measure"].to_numpy()
        acm = d["abs_close_move"].to_numpy(); tv = d["avg_traded_value20_cr"].to_numpy()
        ivd = d["is_volatile_day"].to_numpy(); pvm = d["past_vol_mean"].to_numpy()
        ts = d["trade_stop"].to_numpy()
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
            LO = lo[i + 1:i + 1 + MAXHOLD]; C = c[i + 1:i + 1 + MAXHOLD]
            if len(C) < MAXHOLD or not np.all(np.isfinite(C)): continue
            rec = dict(sym=t, date=d.index[i], typ=typ)
            rec["hold30"], rec["hold30_d"] = (C[-1] / entry - 1) * 100 - COST, MAXHOLD
            for s in (8, 10, 12, 15, 20):
                rec[f"stop{s}"], rec[f"stop{s}_d"] = fixed_stop(entry, LO, C, entry * (1 - s / 100))
            # the stop the scanner already prints on every row
            if np.isfinite(ts[i]) and ts[i] > 0:
                rec["scanner"], rec["scanner_d"] = fixed_stop(entry, LO, C, float(ts[i]))
                rec["scanner_pct"] = (1 - ts[i] / entry) * 100
            else:
                rec["scanner"] = rec["scanner_d"] = rec["scanner_pct"] = np.nan
            rows.append(rec)
    print(f"  chunk {ci}/{len(CH)}: {len(rows):,}", flush=True)
pd.DataFrame(rows).to_csv("fixedstop.csv", index=False)
print(f"DONE {len(rows):,}")
