import pandas as pd, numpy as np
d = pd.read_csv("accel.csv", parse_dates=["date"])
mid = d.date.quantile(.5); A = set(sorted(d.sym.unique())[::2]); alld = sorted(d.date.unique())
B30 = 25.0
d["accel"] = (d.r5 > d.r10) & (d.r10 > d.r20)
d["ema_trend"] = d.ema_stack & (d.s20 > 0) & (d.s50 > 0) & d.above20
print(f"{len(d):,} observations, {d.date.nunique()} dates, {d.sym.nunique()} tickers")
print(f"baseline pool: r15 {d.r15.mean():+.2f}%  r30 {d.r30.mean():+.2f}%  "
      f"P(r30>25%) {np.mean(d.r30>B30)*100:.1f}%\n")

def line(g, lbl, N=None, show_n=True):
    if N is not None:
        g = g[g._r <= N]
    if len(g) < 40:
        print(f"{lbl:<34}  too few rows ({len(g)})"); return
    wins = g.assign(b=g.r30 > B30).groupby("date")["b"].sum().reindex(alld, fill_value=0)
    a = g[g.sym.isin(A)].r30.mean(); b = g[~g.sym.isin(A)].r30.mean()
    t1 = g[g.date <= mid].r30.mean(); t2 = g[g.date > mid].r30.mean()
    names = g.groupby("date").size().reindex(alld, fill_value=0)
    print(f"{lbl:<34}{len(g):>7,}{names.median():>7.0f}{g.r15.mean():>+8.2f}{g.r30.mean():>+8.2f}"
          f"{np.mean(g.r30>B30)*100:>8.1f}%{(wins>=1).mean()*100:>8.1f}%"
          f"{a:>+8.2f}/{b:>+.2f}{t1:>+8.2f}/{t2:>+.2f}")

hdr = (f"{'rule':<34}{'n':>7}{'names':>7}{'r15':>8}{'r30':>8}{'P(big)':>9}{'>=1 win':>9}"
       f"{'  tickA/B':>16}{'  time1/2':>16}")

print("=== 1. ACCELERATION AS A SHAPE (5D > 10D > 20D) ===")
print(hdr)
line(d, "  whole pool (baseline)")
line(d[d.accel], "  accelerating only")
line(d[~d.accel], "  NOT accelerating")

print("\n=== 2. EMA TREND STRUCTURE (20>50, both rising, price above) ===")
print(hdr)
line(d[d.ema_trend], "  full trend structure")
line(d[d.ema_stack], "  EMA20 > EMA50 only")
line(d[d.above20], "  price > EMA20 only")
line(d[~d.ema_trend], "  fails trend structure")

print("\n=== 3. TIGHTER VOLATILITY GATES ===")
print(hdr)
line(d[d.avg_v >= 4.0], "  avg volatility >= 4.0 (was 3.5)")
line(d[d.med_v >= 3.0], "  median volatility >= 3.0 (2.5)")
line(d[d.vdays >= 15], "  volatile days >= 15 (was 12)")
line(d[d.vratio >= 0.25], "  volatility ratio >= 25% (20%)")
line(d[d.spike <= 4.0], "  spike ratio <= 4.0 (was 4.5)")
line(d[(d.avg_v >= 4.0) & (d.med_v >= 3.0) & (d.vdays >= 15) & (d.vratio >= 0.25)],
     "  ALL FOUR tightened")

print("\n=== AS A TOP-20 LIST, ranked by Expected_Move within each rule ===")
print(hdr)
for lbl, sub in [("  current (typ>=18, top 20)", d[d.typ >= 18]),
                 ("  + accelerating", d[(d.typ >= 18) & d.accel]),
                 ("  + EMA trend structure", d[(d.typ >= 18) & d.ema_trend]),
                 ("  + accel AND EMA trend", d[(d.typ >= 18) & d.accel & d.ema_trend]),
                 ("  + all four gates tightened", d[(d.typ >= 18) & (d.avg_v >= 4.0)
                                                    & (d.med_v >= 3.0) & (d.vdays >= 15)
                                                    & (d.vratio >= 0.25)])]:
    s = sub.copy()
    s["_r"] = s.groupby("date")["typ"].rank(ascending=False, method="first")
    line(s, lbl, N=20)
