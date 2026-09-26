# Rank each decision date, keep the top 3, compare exit rules on the SAME trades.
import pandas as pd, numpy as np
d = pd.read_csv("exits.csv", parse_dates=["date"])
d["rank"] = d.groupby("date")["typ"].rank(ascending=False, method="first")
top = d[d["rank"] <= 3].copy()
print(f"{len(top):,} top-3 trades over {top['date'].nunique()} dates "
      f"({top['date'].min().date()} -> {top['date'].max().date()})\n")

RULES = [("owner10","sell once up 10% (you)"),("owner15","sell once up 15% (you)"),
         ("trail8","trailing stop 8%"),("trail12","trailing stop 12%"),
         ("trail15","trailing stop 15%"),("trail20","trailing stop 20%"),
         ("atr20","trailing 2.0x ATR"),("atr30","trailing 3.0x ATR"),
         ("hold30","hold 30 sessions, no stop")]

mid = top["date"].quantile(.5)
tk = sorted(top["sym"].unique()); A = set(tk[::2])
print(f"{'rule':<26}{'avg':>8}{'med':>8}{'win':>7}{'days':>7}{'/day':>7}"
      f"{'  tickerA/B':>16}{'  time1/2':>16}")
print("-" * 95)
for col, label in RULES:
    v = top[col].dropna()
    per_day = v.mean() / top[f'{col}_d'].mean()
    a = top[top["sym"].isin(A)][col].mean(); b = top[~top["sym"].isin(A)][col].mean()
    t1 = top[top["date"] <= mid][col].mean(); t2 = top[top["date"] > mid][col].mean()
    print(f"{label:<26}{v.mean():>+8.2f}{v.median():>+8.2f}{(v>0).mean()*100:>6.1f}%"
          f"{top[f'{col}_d'].mean():>7.1f}{per_day:>+7.2f}"
          f"{a:>+8.2f}/{b:>+.2f}{t1:>+9.2f}/{t2:>+.2f}")

print("\nWhat the trailing stop does to the tails (12% trail vs hold30):")
for lo, hi, name in [(-99,-10,"big losers  <-10%"),(-10,0,"small losers"),
                     (0,10,"small wins"),(10,25,"good wins 10-25%"),(25,999,"big wins >25%")]:
    m = (top["hold30"] >= lo) & (top["hold30"] < hi)
    if m.sum() == 0: continue
    print(f"  {name:<20} n={m.sum():>4}   hold30 {top.loc[m,'hold30'].mean():>+7.2f}"
          f"   trail12 {top.loc[m,'trail12'].mean():>+7.2f}"
          f"   held {top.loc[m,'trail12_d'].mean():>4.1f}d")
