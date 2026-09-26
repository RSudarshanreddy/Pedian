# The three parts of the review that are NOT already measured.
#
# Its diagnosis is right -- Expected_Move is a 60-day historical measure, not
# current momentum. Its prescription mostly is not: all seven Momentum_Strength
# components were ranked today and every one lost to Expected_Move by 1.3-4.5pp.
#
# What survives as untested:
#   1. ACCELERATION AS A SHAPE. I tested r5/r10/r20 as levels. The review asks
#      for the ORDERING -- 5D > 10D > 20D, i.e. +3% -> +8% -> +15% beats
#      +15% -> +8% -> +2%. A level cannot express that and my features could
#      not see it.
#   2. EMA20 > EMA50, BOTH SLOPES POSITIVE. EMA50 is not computed anywhere in
#      momentum.py. Genuinely new: trend structure, not trend distance.
#   3. TIGHTER VOLATILITY GATES. avg 3.5->4.0, median 2.5->3.0, volatile days
#      12->15, ratio 20%->25%. Never swept.
#
# Scored the way the owner actually uses the list: P(r30 > 25%) -- does the
# list contain a winner -- alongside mean, with disjoint ticker and time halves.
import warnings, sys, logging; warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
sys.path.insert(0, "C:/Users/Lenovo/Desktop/stocks/Pedian/scanner/src")
import pandas as pd, numpy as np, yfinance as yf, momentum as M

cfg = M.ScannerConfig(); COST = cfg.round_trip_cost_pct; MH = cfg.move_horizon_days
L = 60; HOR = (7, 15, 30)


def froll(a, w, fn):
    return getattr(pd.Series(a).rolling(w), fn)().shift(-(w - 1)).to_numpy()


def pct_back(a, k):
    out = np.full(len(a), np.nan)
    out[k:] = (a[k:] / a[:-k] - 1) * 100
    return out


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
        vm = d["volatility_measure"].to_numpy(); acm = d["abs_close_move"].to_numpy()
        tv = d["avg_traded_value20_cr"].to_numpy(); ivd = d["is_volatile_day"].to_numpy()
        pvm = d["past_vol_mean"].to_numpy()
        ema20 = d["EMA20"].to_numpy()
        ema50 = d["Close"].ewm(span=50).mean().to_numpy()      # NEW -- not in the file
        n = len(d)

        r5 = pct_back(c, 5); r10 = pct_back(c, 10); r20 = pct_back(c, 20)
        # slopes over 5 sessions, in percent, so they are comparable across prices
        s20 = np.full(n, np.nan); s50 = np.full(n, np.nan)
        s20[5:] = (ema20[5:] / ema20[:-5] - 1) * 100
        s50[5:] = (ema50[5:] / ema50[:-5] - 1) * 100

        pk = (np.append(froll(c, MH, "max")[1:], np.nan) - c) / c * 100 - COST
        elig = pvm >= cfg.min_avg_volatility
        pk_e = np.where(elig, pk, np.nan)

        for i in range(300, n - max(HOR) - 2, 1):
            if not (cfg.min_price < c[i] < cfg.max_price): continue
            if not np.isfinite(tv[i]) or tv[i] < cfg.min_avg_traded_value_cr: continue
            entry = o[i + 1]
            if not np.isfinite(entry) or entry <= 0: continue
            end = i - MH; lb = slice(max(0, end - L), end)
            p = pk_e[lb]; p = p[np.isfinite(p)]
            if len(p) < cfg.min_persistence_sample: continue
            typ = float(np.median(p))
            if typ < cfg.min_typical_move_pct: continue
            hh = vm[max(0, i - L):i]; hh = hh[np.isfinite(hh)]
            if len(hh) < 40: continue
            avg_v = float(np.mean(hh)); med_v = float(np.median(hh))
            if avg_v < cfg.min_avg_volatility or med_v < cfg.min_median_volatility: continue
            vdays = int(ivd[max(0, i - L):i].sum())
            if vdays < cfg.min_volatile_days: continue
            vratio = vdays / len(hh)
            if vratio < cfg.min_volatility_ratio: continue
            mv_ = acm[max(0, i - 180):i]; mv_ = mv_[np.isfinite(mv_)]
            if len(mv_) < 120: continue
            spike = float(np.percentile(mv_, 95)) / max(float(np.median(mv_)), .01)
            if spike > cfg.max_spike_ratio: continue
            w = c[max(0, i - 250):i + 1]; pa = np.maximum.accumulate(w)
            if float(((w - pa) / pa).min() * 100) < cfg.min_drawdown_pct: continue

            rec = dict(sym=t, date=d.index[i], typ=typ,
                       avg_v=avg_v, med_v=med_v, vdays=vdays, vratio=vratio, spike=spike,
                       r5=r5[i], r10=r10[i], r20=r20[i],
                       above20=c[i] > ema20[i], ema_stack=ema20[i] > ema50[i],
                       s20=s20[i], s50=s50[i])
            ok = True
            for H in HOR:
                if not np.isfinite(c[i + H]): ok = False; break
                rec[f"r{H}"] = (c[i + H] / entry - 1) * 100 - COST
            if ok: rows.append(rec)
    print(f"  chunk {ci}/{len(CH)}: {len(rows):,}", flush=True)
pd.DataFrame(rows).to_csv("accel.csv", index=False)
print(f"DONE {len(rows):,}")
