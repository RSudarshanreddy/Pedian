# What made ANTELOPUS/AEGISLOG/SHILPAMED different from GANECOS?
#
# The owner's question, and the right one: not fresh-vs-stale, but what a
# WINNING candidate looks like at the moment the scanner prints it. From the
# live rows for his four names:
#
#   ANTELOPUS  rank 5-15   liquidity 12 -> 375 Cr (31x)  upside 77-80%  +49.8%
#   SHILPAMED  rank 8-17   liquidity 113 -> 129 Cr       upside 76-100% +37.3%
#   AEGISLOG   rank 1-4    liquidity 150 -> 127 Cr       upside 76.7%   profitable
#   GANECOS    rank 27-118 liquidity 15 -> 10 Cr (down)  upside 58-60%  stuck
#
# Two candidate separators, neither of which is Expected_Move: Upside_Dominance
# (share of forward windows where the peak gain beat the worst loss) and
# LIQUIDITY EXPANSION (money flowing in vs its own recent baseline).
#
# Judged on P(big winner), NOT on the mean -- "all that matters is momentum and
# a winning candidate". A ranker that lifts the average by trimming losers is
# not what is being asked for.
import warnings, sys, logging; warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
sys.path.insert(0, "C:/Users/Lenovo/Desktop/stocks/Pedian/scanner/src")
import pandas as pd, numpy as np, yfinance as yf, momentum as M

cfg = M.ScannerConfig(); COST = cfg.round_trip_cost_pct; MH = cfg.move_horizon_days
LB = cfg.persistence_lookback; L = 60; HOR = (7, 15, 30)
print(f"persistence_lookback={LB}  move_horizon={MH}", flush=True)


def froll(a, w, fn):
    return getattr(pd.Series(a).rolling(w), fn)().shift(-(w - 1)).to_numpy()


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
        tv = d["avg_traded_value20_cr"].to_numpy(); vm = d["volatility_measure"].to_numpy()
        acm = d["abs_close_move"].to_numpy(); ivd = d["is_volatile_day"].to_numpy()
        pvm = d["past_vol_mean"].to_numpy(); ema = d["EMA20"].to_numpy()
        n = len(d)

        # forward peak / trough over MH, per day -- the raw material for BOTH
        # Expected_Move and upside dominance, computed once
        fmax = np.append(froll(c, MH, "max")[1:], np.nan)
        fmin = np.append(froll(c, MH, "min")[1:], np.nan)
        peak = (fmax - c) / c * 100 - COST
        trough = (fmin - c) / c * 100 - COST
        elig = pvm >= cfg.min_avg_volatility
        win = (peak > np.abs(trough)).astype(float)
        win[~np.isfinite(peak) | ~np.isfinite(trough) | ~elig] = np.nan

        # trailing means that CLOSE at i-MH, so no lookahead
        up_roll = pd.Series(win).rolling(LB, min_periods=20).mean().to_numpy()
        pk_elig = np.where(elig, peak, np.nan)

        # liquidity expansion: today's 20d traded value vs its own 3-month base
        tv_base = pd.Series(tv).rolling(60).median().to_numpy()

        for i in range(300, n - max(HOR) - 2, 1):
            if not (cfg.min_price < c[i] < cfg.max_price): continue
            if not np.isfinite(tv[i]) or tv[i] < cfg.min_avg_traded_value_cr: continue
            entry = o[i + 1]
            if not np.isfinite(entry) or entry <= 0: continue
            end = i - MH
            lb = slice(max(0, end - L), end)
            p = pk_elig[lb]; p = p[np.isfinite(p)]
            if len(p) < cfg.min_persistence_sample: continue
            typ = float(np.median(p))
            if typ < cfg.min_typical_move_pct: continue
            hh = vm[max(0, i - L):i]; hh = hh[np.isfinite(hh)]
            if len(hh) < 40 or float(np.mean(hh)) < cfg.min_avg_volatility: continue
            if float(np.median(hh)) < cfg.min_median_volatility: continue
            if int(ivd[max(0, i - L):i].sum()) < cfg.min_volatile_days: continue
            mv_ = acm[max(0, i - 180):i]; mv_ = mv_[np.isfinite(mv_)]
            if len(mv_) < 120: continue
            if float(np.percentile(mv_, 95)) / max(float(np.median(mv_)), .01) > cfg.max_spike_ratio: continue
            w = c[max(0, i - 250):i + 1]; pa = np.maximum.accumulate(w)
            if float(((w - pa) / pa).min() * 100) < cfg.min_drawdown_pct: continue

            up = up_roll[end] * 100 if end >= 0 and np.isfinite(up_roll[end]) else np.nan
            liq_x = tv[i] / tv_base[i] if np.isfinite(tv_base[i]) and tv_base[i] > 0 else np.nan
            rec = dict(sym=t, date=d.index[i], typ=typ, upside=up, liq_x=liq_x,
                       liq=tv[i], dist_ema=(c[i]/ema[i]-1)*100 if np.isfinite(ema[i]) else np.nan)
            ok = True
            for H in HOR:
                if not np.isfinite(c[i + H]): ok = False; break
                rec[f"r{H}"] = (c[i + H] / entry - 1) * 100 - COST
            if ok: rows.append(rec)
    print(f"  chunk {ci}/{len(CH)}: {len(rows):,}", flush=True)
pd.DataFrame(rows).to_csv("winners.csv", index=False)
print(f"DONE {len(rows):,}")
