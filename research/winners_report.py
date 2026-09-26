import pandas as pd, numpy as np
d = pd.read_csv("winners.csv", parse_dates=["date"]).dropna(subset=["upside","liq_x"])
mid = d.date.quantile(.5); A = set(sorted(d.sym.unique())[::2])
BIG15, BIG30 = 15.0, 25.0
print(f"{len(d):,} observations, {d.date.nunique()} dates, {d.sym.nunique()} tickers")
print(f"base rates: P(r15>{BIG15}%)={np.mean(d.r15>BIG15)*100:.1f}%  "
      f"P(r30>{BIG30}%)={np.mean(d.r30>BIG30)*100:.1f}%  mean r15 {d.r15.mean():+.2f}%\n")

def band(col, qs=(0,.2,.4,.6,.8,1.0)):
    e = d[col].quantile(list(qs)).values
    return pd.cut(d[col], bins=np.unique(e), include_lowest=True, duplicates="drop")

for col, nice in [("upside","Upside_Dominance %"), ("liq_x","liquidity vs own 3mo base"),
                  ("typ","Expected_Move %"), ("liq","traded value Cr")]:
    print(f"=== {nice} ===")
    d["_b"] = band(col)
    print(f"{'band':<22}{'n':>6}{'mean r15':>10}{'P(r15>15)':>11}{'P(r30>25)':>11}"
          f"{'  tickA/B':>16}{'  time1/2':>16}")
    for b, g in d.groupby("_b", observed=True):
        a1 = g[g.sym.isin(A)].r15.mean(); b1 = g[~g.sym.isin(A)].r15.mean()
        t1 = g[g.date<=mid].r15.mean(); t2 = g[g.date>mid].r15.mean()
        print(f"{str(b):<22}{len(g):>6,}{g.r15.mean():>+10.2f}{np.mean(g.r15>BIG15)*100:>10.1f}%"
              f"{np.mean(g.r30>BIG30)*100:>10.1f}%{a1:>+8.2f}/{b1:>+.2f}{t1:>+8.2f}/{t2:>+.2f}")
    print()

print("=" * 100)
print("AS A RANKER -- top 3 per session, entry next open")
print("=" * 100)
out = []
def ev(s, lbl, N=3):
    g = s[s._r <= N]
    if len(g) < 60: return
    a1 = g[g.sym.isin(A)].r15.mean(); b1 = g[~g.sym.isin(A)].r15.mean()
    t1 = g[g.date<=mid].r15.mean(); t2 = g[g.date>mid].r15.mean()
    out.append(dict(ranker=lbl, n=len(g), r7=g.r7.mean(), r15=g.r15.mean(), r30=g.r30.mean(),
                    big=np.mean(g.r15>BIG15)*100, big30=np.mean(g.r30>BIG30)*100,
                    tA=a1, tB=b1, h1=t1, h2=t2))
for col, lbl in [("typ","Expected_Move (current)"),("upside","Upside_Dominance"),
                 ("liq_x","liquidity expansion"),("liq","traded value")]:
    s = d.copy(); s["_r"] = s.groupby("date")[col].rank(ascending=False, method="first"); ev(s, lbl)
for f, lbl in [("upside","typ + upside"),("liq_x","typ + liq_x"),("dist_ema","typ + dist_ema")]:
    s = d.dropna(subset=[f]).copy()
    s["_z"] = s.groupby("date")["typ"].rank(pct=True) + s.groupby("date")[f].rank(pct=True)
    s["_r"] = s.groupby("date")["_z"].rank(ascending=False, method="first"); ev(s, lbl)
for thr in (60, 65, 70, 75):
    s = d[d.upside >= thr].copy()
    s["_r"] = s.groupby("date")["typ"].rank(ascending=False, method="first")
    ev(s, f"typ, upside>={thr}%")
for thr in (1.2, 1.5, 2.0):
    s = d[d.liq_x >= thr].copy()
    s["_r"] = s.groupby("date")["typ"].rank(ascending=False, method="first")
    ev(s, f"typ, liq_x>={thr}")
s = d[(d.upside >= 65) & (d.liq_x >= 1.2)].copy()
s["_r"] = s.groupby("date")["typ"].rank(ascending=False, method="first")
ev(s, "typ, upside>=65 & liq_x>=1.2")

r = pd.DataFrame(out).sort_values("big", ascending=False)
print(f"{'ranker':<30}{'n':>6}{'r7':>7}{'r15':>8}{'r30':>8}{'P(big15)':>10}{'P(big30)':>10}"
      f"{'  tickA/B':>16}{'  time1/2':>16}")
for x in r.itertuples():
    star = " <<<" if "current" in x.ranker else ""
    print(f"{x.ranker:<30}{x.n:>6}{x.r7:>+7.2f}{x.r15:>+8.2f}{x.r30:>+8.2f}"
          f"{x.big:>9.1f}%{x.big30:>9.1f}%{x.tA:>+8.2f}/{x.tB:>+.2f}{x.h1:>+8.2f}/{x.h2:>+.2f}{star}")
