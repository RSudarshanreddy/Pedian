# Does FRESHNESS predict? -- the one measurement before touching the digest.
#
# The owner's complaint is that the top 3 is frozen for days: KABRAEXTRU/SETL/
# INDSWFTLAB were rank 1-2-3 for four straight sessions, PANAMAPET was rank 1 on
# 12 of 19. Expected_Move is a median over 60 sessions, so it moves ~1.7% of its
# mass per day -- a near-constant, so a near-constant list.
#
# Setup_Age_Days already measures the missing dimension and is thrown away at
# the sort. Before surfacing it I have to know whether it predicts, because an
# EARLIER measurement in this project cuts the other way: within the top 3,
# Action=BUY returned +1.36% against WATCH's +3.71%, and `coiling` was the best
# setup at +5.32%. If that holds pool-wide, a "FRESH TODAY" block that prefers
# BUY/breakout is noise dressed up as a signal.
#
# Measured at 7 and 15 sessions, NOT 30 -- the owner's real round trips average
# 10 calendar days (~7 sessions) at +6.68%, so that is the horizon that matters.
import warnings, sys, logging; warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
sys.path.insert(0, "C:/Users/Lenovo/Desktop/stocks/Pedian/scanner/src")
import pandas as pd, numpy as np, yfinance as yf, momentum as M

cfg = M.ScannerConfig(); COST = cfg.round_trip_cost_pct; MH = cfg.move_horizon_days
L = 60; STEP = 1; HORIZONS = (7, 15)


def setup_vec(d):
    """Vectorised get_entry_trigger. Same branch order, same thresholds."""
    c = d["Close"].to_numpy(); o = d["Open"].to_numpy(); h = d["High"].to_numpy()
    ema = d["EMA20"].to_numpy(); pb = d["pullback_pct"].to_numpy()
    h20p = d["high_20_prev"].to_numpy(); vs = d["volume_spike"].to_numpy()
    n = len(d)
    pc = np.concatenate([[np.nan], c[:-1]])
    ph = np.concatenate([[np.nan], h[:-1]])
    pe = np.concatenate([[np.nan], ema[:-1]])
    bullish = c > o
    bvo = np.where(np.isfinite(vs), vs >= cfg.breakout_volume_mult, True)
    conds = [
        (c > h20p) & bvo & bullish,
        (pb >= cfg.pullback_min) & (pb <= cfg.pullback_max) & (c > ema) & bullish & (c > ph),
        (c > ema) & (pc <= pe) & bullish,
        pb > 12,
        pb < cfg.pullback_min,
        c < ema,
    ]
    names = ["breakout", "pullback_bounce", "reclaim", "deep_pullback", "extended", "below_trend"]
    st = np.select(conds, names, default="coiling").astype(object)
    bad = ~(np.isfinite(ema) & np.isfinite(h20p) & np.isfinite(pb))
    st[bad] = "no_data"
    st[0] = "no_data"
    return st


def run_length(st):
    """age[i] = how many consecutive sessions ending at i share st[i]."""
    age = np.ones(len(st), dtype=int)
    for i in range(1, len(st)):
        if st[i] == st[i - 1]:
            age[i] = age[i - 1] + 1
    return age


def froll(a, w, fn):
    return getattr(pd.Series(a).rolling(w), fn)().shift(-(w - 1)).to_numpy()


# ---- verify the vectorised version against the real one -------------------
VERIFIED = {"ok": 0, "bad": 0}


def verify(d, st):
    for k in np.random.default_rng(0).integers(300, len(d), size=12):
        _, real, _ = M.get_entry_trigger(d.iloc[:int(k) + 1], cfg)
        if real == st[int(k)]:
            VERIFIED["ok"] += 1
        else:
            VERIFIED["bad"] += 1
            if VERIFIED["bad"] <= 5:
                print(f"    MISMATCH at {k}: real={real} vec={st[int(k)]}", flush=True)


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
        st = setup_vec(d); age = run_length(st)
        if ci <= 2: verify(d, st)
        c = d["Close"].to_numpy(); o = d["Open"].to_numpy()
        vm = d["volatility_measure"].to_numpy(); acm = d["abs_close_move"].to_numpy()
        tv = d["avg_traded_value20_cr"].to_numpy(); ivd = d["is_volatile_day"].to_numpy()
        pvm = d["past_vol_mean"].to_numpy()
        n = len(d)
        pk = (np.append(froll(c, MH, "max")[1:], np.nan) - c) / c * 100 - COST
        elig = pvm >= cfg.min_avg_volatility
        for i in range(300, n - max(HORIZONS) - 2, STEP):
            if st[i] in ("no_data",): continue
            px = c[i]
            if not (cfg.min_price < px < cfg.max_price): continue
            if not np.isfinite(tv[i]) or tv[i] < cfg.min_avg_traded_value_cr: continue
            entry = o[i + 1]
            if not np.isfinite(entry) or entry <= 0: continue
            end = i - MH; lb = slice(max(0, end - L), end)
            p = pk[lb][elig[lb]]; p = p[np.isfinite(p)]
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
            rec = dict(sym=t, date=d.index[i], typ=typ, setup=st[i], age=int(age[i]))
            ok = True
            for H in HORIZONS:
                ex = c[i + H]
                if not np.isfinite(ex): ok = False; break
                rec[f"r{H}"] = (ex / entry - 1) * 100 - COST
            if ok: rows.append(rec)
    print(f"  chunk {ci}/{len(CH)}: {len(rows):,}", flush=True)

print(f"\nvectoriser check: {VERIFIED['ok']} match, {VERIFIED['bad']} mismatch")
pd.DataFrame(rows).to_csv("fresh.csv", index=False)
print(f"DONE {len(rows):,}")
