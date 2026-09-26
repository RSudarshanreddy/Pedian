import pandas as pd, numpy as np
d = pd.read_csv("rank.csv", parse_dates=["date"])
mid = d.date.quantile(.5); tks = sorted(d.sym.unique()); A = set(tks[::2])
FEATS = ["typ","pullback","vspike","rvol5","dist_ema","r1","r3","r5","r10",
         "range_pct","atr_pct","vol_heat","pos_in_range"]

print(f"{len(d):,} observations, {d.date.nunique()} dates, {d.sym.nunique()} tickers\n")
print("STEP 1 -- does the feature correlate with forward return at all?")
print(f"{'feature':<14}{'corr r7':>10}{'corr r15':>10}   (Spearman, within-date demeaned)")
dm = d.copy()
for f in FEATS + ["r7","r15"]:
    dm[f] = dm.groupby("date")[f].transform(lambda s: s.rank(pct=True))
for f in FEATS:
    c7 = dm[[f,"r7"]].corr(method="spearman").iloc[0,1]
    c15 = dm[[f,"r15"]].corr(method="spearman").iloc[0,1]
    print(f"{f:<14}{c7:>+10.4f}{c15:>+10.4f}")

def evaluate(sub, col, label, N, out):
    g = sub[sub[f"_r"] <= N]
    if len(g) < 40: return
    a = g[g.sym.isin(A)][col].mean(); b = g[~g.sym.isin(A)][col].mean()
    t1 = g[g.date<=mid][col].mean(); t2 = g[g.date>mid][col].mean()
    sets = g.groupby("date")["sym"].apply(frozenset).sort_index()
    carry = np.mean([len(x & y) for x, y in zip(sets.values[:-1], sets.values[1:])]) if len(sets) > 1 else np.nan
    ident = np.mean([len(x & y) == N for x, y in zip(sets.values[:-1], sets.values[1:])])*100 if len(sets) > 1 else np.nan
    out.append(dict(ranker=label, n=len(g), avg=g[col].mean(), win=(g[col]>0).mean()*100,
                    tA=a, tB=b, h1=t1, h2=t2, carry=carry, ident=ident))

for H in (7, 15):
    col = f"r{H}"
    for N in (2, 3):
        out = []
        for f in FEATS:
            for sign, tag in ((-1, "high"), (1, "low")):
                s = d.dropna(subset=[f]).copy()
                s["_r"] = s.groupby("date")[f].rank(ascending=(sign>0), method="first")
                evaluate(s, col, f"{tag} {f}", N, out)
        # composites: slow capability x daily state
        for f in ["r5","r10","vol_heat","rvol5","pos_in_range","dist_ema"]:
            s = d.dropna(subset=[f,"typ"]).copy()
            s["_z"] = (s.groupby("date")["typ"].rank(pct=True)
                       + s.groupby("date")[f].rank(pct=True))
            s["_r"] = s.groupby("date")["_z"].rank(ascending=False, method="first")
            evaluate(s, col, f"typ + {f}", N, out)
        # setup-aware: Expected_Move but only inside the better setups
        for keep, tag in [(("coiling",),"coiling only"),
                          (("coiling","below_trend"),"coiling+below"),
                          (("coiling","extended"),"coiling+extended"),
                          (("coiling","below_trend","reclaim","extended"),"top4 setups")]:
            s = d[d.setup.isin(keep)].copy()
            s["_r"] = s.groupby("date")["typ"].rank(ascending=False, method="first")
            evaluate(s, col, f"typ, {tag}", N, out)
        s = d[d.age <= 3].copy()
        s["_r"] = s.groupby("date")["typ"].rank(ascending=False, method="first")
        evaluate(s, col, "typ, age<=3", N, out)

        r = pd.DataFrame(out).sort_values("avg", ascending=False)
        print(f"\n{'='*104}\nTOP {N} @ {H} SESSIONS  (baseline = 'high typ', today's scanner)\n{'='*104}")
        print(f"{'ranker':<22}{'n':>6}{'avg':>8}{'win':>7}{'tickA/B':>16}{'time1/2':>16}"
              f"{'carry':>8}{'ident':>8}")
        base = r[r.ranker=="high typ"]
        for x in r.head(12).itertuples():
            star = " <<<" if x.ranker == "high typ" else ""
            print(f"{x.ranker:<22}{x.n:>6}{x.avg:>+8.2f}{x.win:>6.1f}%"
                  f"{x.tA:>+8.2f}/{x.tB:>+.2f}{x.h1:>+8.2f}/{x.h2:>+.2f}"
                  f"{x.carry:>8.2f}{x.ident:>7.0f}%{star}")
        if len(base) and base.index[0] not in r.head(12).index:
            x = base.iloc[0]
            print(f"  ... baseline at rank {list(r.ranker).index('high typ')+1}: "
                  f"{x.avg:+.2f}%  carry {x.carry:.2f}  ident {x.ident:.0f}%")
