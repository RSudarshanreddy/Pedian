# Can the ranking be made DIRECT -- ranked on what actually predicts, and on
# something that CHANGES DAILY -- instead of on Expected_Move?
#
# The owner's objection, which is correct: Expected_Move is a 60-session median,
# so it is near-constant (KABRAEXTRU 69.75 -> 69.75 -> 69.75 -> 69.88 over four
# days) and the list is therefore near-constant: 64% of sessions are FULLY
# identical to the previous one, 2.60 of 3 names carried over. Meanwhile setup
# state is recomputed every day, carries measured signal, and is used only as a
# tiebreaker. Score is stored but line 35 records that it does not predict.
#
# So: dump every DAILY-CHANGING feature the scanner already computes, plus the
# slow one, and let a sweep decide. Judged on three things at once --
#   (a) does the top 2-3 still earn,
#   (b) does the list actually turn over,
#   (c) do disjoint ticker halves and time halves agree.
# A ranker that is fresh but earns nothing is not a fix.
import warnings, sys, logging; warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
sys.path.insert(0, "C:/Users/Lenovo/Desktop/stocks/Pedian/scanner/src")
import pandas as pd, numpy as np, yfinance as yf, momentum as M

cfg = M.ScannerConfig(); COST = cfg.round_trip_cost_pct; MH = cfg.move_horizon_days
L = 60; HORIZONS = (7, 15)


def setup_vec(d):
    c = d["Close"].to_numpy(); o = d["Open"].to_numpy(); h = d["High"].to_numpy()
    ema = d["EMA20"].to_numpy(); pb = d["pullback_pct"].to_numpy()
    h20p = d["high_20_prev"].to_numpy(); vs = d["volume_spike"].to_numpy()
    pc = np.concatenate([[np.nan], c[:-1]]); ph = np.concatenate([[np.nan], h[:-1]])
    pe = np.concatenate([[np.nan], ema[:-1]])
    bull = c > o
    bvo = np.where(np.isfinite(vs), vs >= cfg.breakout_volume_mult, True)
    conds = [(c > h20p) & bvo & bull,
             (pb >= cfg.pullback_min) & (pb <= cfg.pullback_max) & (c > ema) & bull & (c > ph),
             (c > ema) & (pc <= pe) & bull,
             pb > 12, pb < cfg.pullback_min, c < ema]
    st = np.select(conds, ["breakout","pullback_bounce","reclaim","deep_pullback",
                           "extended","below_trend"], default="coiling").astype(object)
    st[~(np.isfinite(ema) & np.isfinite(h20p) & np.isfinite(pb))] = "no_data"
    st[0] = "no_data"
    return st


def froll(a, w, fn):
    return getattr(pd.Series(a).rolling(w), fn)().shift(-(w - 1)).to_numpy()


def pct_change(a, k):
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
        st = setup_vec(d)
        age = np.ones(len(st), dtype=int)
        for i in range(1, len(st)):
            if st[i] == st[i-1]: age[i] = age[i-1] + 1
        c = d["Close"].to_numpy(); o = d["Open"].to_numpy()
        ema = d["EMA20"].to_numpy(); atr = d["ATR"].to_numpy()
        pb = d["pullback_pct"].to_numpy(); vs = d["volume_spike"].to_numpy()
        rp = d["range_pct"].to_numpy(); vm = d["volatility_measure"].to_numpy()
        acm = d["abs_close_move"].to_numpy(); tv = d["avg_traded_value20_cr"].to_numpy()
        ivd = d["is_volatile_day"].to_numpy(); pvm = d["past_vol_mean"].to_numpy()
        h20 = d["high_20"].to_numpy(); rlo = d["recent_low"].to_numpy()
        n = len(d)
        r1 = pct_change(c,1); r3 = pct_change(c,3); r5 = pct_change(c,5); r10 = pct_change(c,10)
        rvol5 = pd.Series(vs).rolling(5).mean().to_numpy()
        vm60 = pd.Series(vm).rolling(60).mean().to_numpy()
        pk = (np.append(froll(c, MH, "max")[1:], np.nan) - c) / c * 100 - COST
        elig = pvm >= cfg.min_avg_volatility
        for i in range(300, n - max(HORIZONS) - 2, 1):
            if st[i] == "no_data": continue
            if not (cfg.min_price < c[i] < cfg.max_price): continue
            if not np.isfinite(tv[i]) or tv[i] < cfg.min_avg_traded_value_cr: continue
            entry = o[i+1]
            if not np.isfinite(entry) or entry <= 0: continue
            end = i - MH; lb = slice(max(0, end - L), end)
            p = pk[lb][elig[lb]]; p = p[np.isfinite(p)]
            if len(p) < cfg.min_persistence_sample: continue
            typ = float(np.median(p))
            if typ < cfg.min_typical_move_pct: continue
            hh = vm[max(0,i-L):i]; hh = hh[np.isfinite(hh)]
            if len(hh) < 40 or float(np.mean(hh)) < cfg.min_avg_volatility: continue
            if float(np.median(hh)) < cfg.min_median_volatility: continue
            if int(ivd[max(0,i-L):i].sum()) < cfg.min_volatile_days: continue
            mv_ = acm[max(0,i-180):i]; mv_ = mv_[np.isfinite(mv_)]
            if len(mv_) < 120: continue
            if float(np.percentile(mv_,95))/max(float(np.median(mv_)),.01) > cfg.max_spike_ratio: continue
            w = c[max(0,i-250):i+1]; pa = np.maximum.accumulate(w)
            if float(((w-pa)/pa).min()*100) < cfg.min_drawdown_pct: continue
            rec = dict(sym=t, date=d.index[i], setup=st[i], age=int(age[i]),
                       typ=typ,                                    # the SLOW one
                       pullback=pb[i], vspike=vs[i], rvol5=rvol5[i],
                       dist_ema=(c[i]/ema[i]-1)*100 if np.isfinite(ema[i]) else np.nan,
                       r1=r1[i], r3=r3[i], r5=r5[i], r10=r10[i],
                       range_pct=rp[i], atr_pct=atr[i]/c[i]*100 if np.isfinite(atr[i]) else np.nan,
                       vol_heat=vm[i]/vm60[i] if np.isfinite(vm60[i]) and vm60[i] > 0 else np.nan,
                       pos_in_range=((c[i]-rlo[i])/(h20[i]-rlo[i])*100
                                     if np.isfinite(h20[i]) and np.isfinite(rlo[i]) and h20[i] > rlo[i] else np.nan))
            ok = True
            for H in HORIZONS:
                if not np.isfinite(c[i+H]): ok = False; break
                rec[f"r{H}"] = (c[i+H]/entry - 1)*100 - COST
            if ok: rows.append(rec)
    print(f"  chunk {ci}/{len(CH)}: {len(rows):,}", flush=True)
pd.DataFrame(rows).to_csv("rank.csv", index=False)
print(f"DONE {len(rows):,}")
