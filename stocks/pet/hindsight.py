"""Did the dips move up and then cool down? And would selling sooner have fixed it?

Replays the same entries as backtest.py with 15 different exits. The best one
is picked WITH hindsight, so it is an upper bound: if even that loses to a fund,
no exit rule rescues these entries. A winning cell would prove nothing.
"""
import numpy as np, pandas as pd, data, rules, backtest

bars = data.load_bars(); r = rules.Rules()
t = backtest.all_trades(bars, r)
t = t[t.reason != "open"].reset_index(drop=True)

def replay(tr, tp_pct, max_days):
    df = bars[tr.symbol]; i0 = df.index.get_loc(tr.entry_date)
    o, h, l, c = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    tp = tr.entry * (1 + tp_pct / 100) if tp_pct else tr.target
    pend = c[i0] < tr.stop
    for j in range(i0 + 1, len(df)):
        d = j - i0 + 1
        if pend: return df.index[j], o[j]
        if o[j] >= tp: return df.index[j], o[j]
        if h[j] >= tp: return df.index[j], tp
        if d >= max_days: return df.index[j], c[j]
        pend = c[j] < tr.stop
    return df.index[-1], c[-1]

# 1. how far up did each trade get before it ended (max favourable excursion)?
mfe = []
for tr in t.itertuples():
    df = bars[tr.symbol]; i0 = df.index.get_loc(tr.entry_date); i1 = df.index.get_loc(tr.exit_date)
    mfe.append((df.High.iloc[i0 + 1:i1 + 1].max() / tr.entry - 1) * 100 if i1 > i0 else 0.0)
t["mfe"] = mfe
print("HOW HIGH DID IT GO AFTER BUYING, before the trade ended?  (% of trades)")
rows = {}
for reason, g in t.groupby("reason"):
    rows[reason] = {f">= +{x}%": (g.mfe >= x).mean() * 100 for x in (2, 3, 5, 7)}
print(pd.DataFrame(rows).T.round(0).to_string(), "\n")

# 2. every exit in a grid; the best one is chosen with hindsight = an upper bound
res = {}
for tp in (None, 3, 5, 7, 10):
    for days in (5, 10, 15):
        x = t.copy()
        ex = [replay(tr, tp, days) for tr in x.itertuples()]
        x["exit_date"] = [e[0] for e in ex]; x["exit"] = [e[1] for e in ex]
        x["ret"] = (x.exit / x.entry - 1) * 100 - r.cost_pct
        eq = backtest.portfolio(x, bars, r.cost_pct).equity
        cs = backtest.curve_stats(eq)
        res[f"{'range 3/4' if tp is None else f'+{tp}%'} / {days}d"] = {
            "avg trade %": x.ret.mean(), "win %": (x.ret > 0).mean() * 100,
            "account CAGR %": cs["CAGR %"], "worst drawdown %": cs["worst drawdown %"]}
print("EVERY EXIT TRIED -- sell target / max days held (same entries, same stop)")
print(pd.DataFrame(res).T.round(2).sort_values("account CAGR %", ascending=False).to_string())
