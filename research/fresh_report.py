import pandas as pd, numpy as np
d = pd.read_csv("fresh.csv", parse_dates=["date"])
mid = d["date"].quantile(.5); tk = sorted(d["sym"].unique()); A = set(tk[::2])

def split(sub, col):
    a = sub[sub.sym.isin(A)][col].mean(); b = sub[~sub.sym.isin(A)][col].mean()
    t1 = sub[sub.date <= mid][col].mean(); t2 = sub[sub.date > mid][col].mean()
    return a, b, t1, t2

print(f"{len(d):,} qualifying observations, {d.date.nunique()} dates, "
      f"{d.sym.nunique()} tickers\n")

for H in (7, 15):
    col = f"r{H}"
    print(f"=== {H} SESSIONS -- by setup type ===")
    print(f"{'setup':<17}{'n':>7}{'avg':>8}{'med':>8}{'win':>7}{'  tickerA/B':>16}{'  time1/2':>16}")
    for s, g in sorted(d.groupby("setup"), key=lambda x: -x[1][col].mean()):
        if len(g) < 60: continue
        a, b, t1, t2 = split(g, col)
        print(f"{s:<17}{len(g):>7,}{g[col].mean():>+8.2f}{g[col].median():>+8.2f}"
              f"{(g[col]>0).mean()*100:>6.1f}%{a:>+8.2f}/{b:>+.2f}{t1:>+9.2f}/{t2:>+.2f}")
    print(f"\n--- fresh (age==1) vs stale, {H} sessions ---")
    for lbl, m in [("age == 1 (triggered today)", d.age == 1),
                   ("age 2-3", d.age.between(2, 3)),
                   ("age 4-7", d.age.between(4, 7)),
                   ("age 8+", d.age >= 8)]:
        g = d[m]; a, b, t1, t2 = split(g, col)
        print(f"  {lbl:<28}{len(g):>7,}{g[col].mean():>+8.2f}{(g[col]>0).mean()*100:>6.1f}%"
              f"{a:>+8.2f}/{b:>+.2f}{t1:>+9.2f}/{t2:>+.2f}")
    print()

# ---- the decisive test: the two digest blocks, head to head ----------------
print("=" * 78)
print("THE TWO BLOCKS, same dates, 3 names each, entry next open")
print("=" * 78)
d["rank_move"] = d.groupby("date")["typ"].rank(ascending=False, method="first")
FRESH_SETUPS = ("breakout", "reclaim")
d["is_fresh"] = (d.age == 1) & d.setup.isin(FRESH_SETUPS)
f = d[d.is_fresh].copy()
f["rank_fresh"] = f.groupby("date")["typ"].rank(ascending=False, method="first")

for H in (7, 15):
    col = f"r{H}"
    print(f"\n-- {H} sessions --")
    blocks = {
        "STRONGEST top3 (today's digest)": d[d.rank_move <= 3],
        "STRONGEST top3, no `extended`": (d[(d.setup != "extended")]
            .assign(r2=lambda x: x.groupby("date")["typ"].rank(ascending=False, method="first"))
            .query("r2 <= 3")),
        "FRESH top3 (age1, breakout/reclaim)": f[f.rank_fresh <= 3],
        "all candidates pooled": d,
    }
    for lbl, g in blocks.items():
        if not len(g): continue
        a, b, t1, t2 = split(g, col)
        print(f"  {lbl:<38}{len(g):>6,}{g[col].mean():>+8.2f}{(g[col]>0).mean()*100:>6.1f}%"
              f"{a:>+8.2f}/{b:>+.2f}{t1:>+9.2f}/{t2:>+.2f}")
    cov = f.date.nunique() / d.date.nunique() * 100
    print(f"  FRESH block has candidates on {cov:.0f}% of dates "
          f"(median {f.groupby('date').size().median():.0f} per date)")

# ---- would the FRESH block actually change day to day? --------------------
print("\n--- turnover: how often does the top 3 change? ---")
for lbl, g, rc in [("STRONGEST", d[d.rank_move <= 3], "rank_move"),
                   ("FRESH", f[f.rank_fresh <= 3], "rank_fresh")]:
    sets = g.groupby("date")["sym"].apply(frozenset).sort_index()
    same = [len(a & b) for a, b in zip(sets.values[:-1], sets.values[1:])]
    print(f"  {lbl:<11}{len(sets)} dates, avg {np.mean(same):.2f} of 3 names "
          f"carried over from the previous session "
          f"({np.mean([s==3 for s in same])*100:.0f}% fully identical)")
