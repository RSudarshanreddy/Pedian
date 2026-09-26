import pandas as pd, numpy as np
d = pd.read_csv("fixedstop.csv", parse_dates=["date"])
d["rank"] = d.groupby("date")["typ"].rank(ascending=False, method="first")
top = d[d["rank"] <= 3].copy()
print(f"{len(top):,} top-3 trades, {top['date'].nunique()} dates\n")
RULES = [("hold30","no stop at all"),("stop8","fixed stop -8%"),("stop10","fixed stop -10%"),
         ("stop12","fixed stop -12%"),("stop15","fixed stop -15%"),("stop20","fixed stop -20%"),
         ("scanner","scanner's own Stop_Loss")]
mid = top["date"].quantile(.5); tk = sorted(top["sym"].unique()); A = set(tk[::2])
print(f"{'rule':<26}{'avg':>8}{'med':>8}{'win':>7}{'days':>7}{'stopped':>9}"
      f"{'  tickerA/B':>16}{'  time1/2':>16}")
print("-" * 97)
for col, label in RULES:
    v = top[col].dropna()
    if not len(v): continue
    hit = (top[f"{col}_d"] < 30).mean() * 100
    a = top[top["sym"].isin(A)][col].mean(); b = top[~top["sym"].isin(A)][col].mean()
    t1 = top[top["date"] <= mid][col].mean(); t2 = top[top["date"] > mid][col].mean()
    print(f"{label:<26}{v.mean():>+8.2f}{v.median():>+8.2f}{(v>0).mean()*100:>6.1f}%"
          f"{top[f'{col}_d'].mean():>7.1f}{hit:>8.1f}%"
          f"{a:>+8.2f}/{b:>+.2f}{t1:>+9.2f}/{t2:>+.2f}")
print(f"\nscanner stop sits {top['scanner_pct'].median():.1f}% below entry (median)")
best = "stop15"
print(f"\nTails, {best} vs no stop:")
for lo, hi, nm in [(-99,-10,"big losers <-10%"),(-10,0,"small losers"),(0,10,"small wins"),
                   (10,25,"good wins 10-25%"),(25,999,"big wins >25%")]:
    m = (top["hold30"] >= lo) & (top["hold30"] < hi)
    if not m.sum(): continue
    print(f"  {nm:<20} n={m.sum():>4}   none {top.loc[m,'hold30'].mean():>+7.2f}"
          f"   {best} {top.loc[m,best].mean():>+7.2f}"
          f"   stopped {(top.loc[m,best+'_d']<30).mean()*100:>5.1f}%")
