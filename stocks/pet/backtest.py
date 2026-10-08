"""Replays rules.py over every NSE stock: trade statistics, splits, and a portfolio.

The trades are simulated per stock as if money were unlimited. The portfolio
then takes them in date order into SLOTS equal slots. When more stocks fill on
the same day than there are free slots, the most actively traded go first.
"""

from __future__ import annotations

import argparse

import pandas as pd
import yfinance as yf

import data
import rules

SLOTS = 10
BENCHMARKS = {"Nifty 50": "^NSEI", "Nifty 200 Momentum 30 ETF": "MOMENTUM.NS"}


def all_trades(bars: dict[str, pd.DataFrame], r: rules.Rules) -> pd.DataFrame:
    rows = [{"symbol": s, **tr} for s, df in bars.items() for tr in rules.trades(df, r)]
    return pd.DataFrame(rows)


def stats(t: pd.DataFrame) -> dict:
    win = t.ret > 0
    return {
        "trades": len(t),
        "win %": win.mean() * 100,
        "avg %": t.ret.mean(),
        "median %": t.ret.median(),
        "avg win %": t.ret[win].mean(),
        "avg loss %": t.ret[~win].mean(),
        "worst %": t.ret.min(),
        "days held": t.days.mean(),
    }


def table(rows: dict[str, dict]) -> str:
    return pd.DataFrame(rows).T.round(2).to_string()


def curve_stats(equity: pd.Series) -> dict:
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    return {
        "CAGR %": ((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) * 100,
        "total %": (equity.iloc[-1] / equity.iloc[0] - 1) * 100,
        "worst drawdown %": (equity / equity.cummax() - 1).min() * 100,
    }


def portfolio(t: pd.DataFrame, bars: dict[str, pd.DataFrame], cost_pct: float) -> pd.DataFrame:
    """Daily equity of a SLOTS-slot account starting at 1.0 on the first entry date."""
    t = t.sort_values(["entry_date", "traded_cr"], ascending=[True, False])
    start = t.entry_date.min()
    calendar = sorted(d for d in set().union(*(df.index for df in bars.values())) if d >= start)
    entries = {d: g for d, g in t.groupby("entry_date")}
    half = cost_pct / 200  # half the round-trip cost on each side
    cash, equity, held, rows = 1.0, 1.0, [], []

    def settle(d: pd.Timestamp) -> None:
        nonlocal cash, held
        cash += sum(p["shares"] * p["exit"] * (1 - half) for p in held if p["exit_date"] == d)
        held = [p for p in held if p["exit_date"] != d]

    for d in calendar:
        settle(d)
        for tr in entries.get(d, pd.DataFrame()).itertuples():
            if len(held) >= SLOTS:
                break
            alloc = min(cash, equity / SLOTS)
            cash -= alloc
            held.append({"symbol": tr.symbol, "exit_date": tr.exit_date, "exit": tr.exit,
                         "shares": alloc * (1 - half) / tr.entry})
        settle(d)  # bought and sold on the same day; otherwise it would be held forever
        equity =cash + sum(p["shares"] * bars[p["symbol"]]["Close"].asof(d) for p in held)
        rows.append((d, equity, len(held)))
    return pd.DataFrame(rows, columns=["date", "equity", "held"]).set_index("date")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true", help="download fresh prices first")
    ap.add_argument("--symbol", help="also list every trade in this stock")
    args = ap.parse_args()

    r = rules.Rules()
    bars = data.load_bars(refresh=args.refresh)
    t = all_trades(bars, r)
    closed = t[t.reason != "open"]
    first = closed.entry_date.min().date()
    print(f"\n{len(bars)} stocks with data, {len(t)} trades ({len(t) - len(closed)} still open), "
          f"entries {first} to {closed.entry_date.max().date()}\n")

    print("ALL CLOSED TRADES (returns after costs)")
    print(table({"all": stats(closed)}), "\n")

    print("BY HOW THEY ENDED")
    print(table({k: stats(g) for k, g in closed.groupby("reason")}), "\n")

    mid = closed.entry_date.sort_values().iloc[len(closed) // 2]
    syms = sorted(bars)
    even = set(syms[::2])
    print("SPLITS -- a real effect should show in every row")
    print(table({
        f"first half (to {mid.date()})": stats(closed[closed.entry_date < mid]),
        "second half": stats(closed[closed.entry_date >= mid]),
        "stocks A (alternate)": stats(closed[closed.symbol.isin(even)]),
        "stocks B (the rest)": stats(closed[~closed.symbol.isin(even)]),
        **{f"year {y}": stats(g) for y, g in closed.groupby(closed.entry_date.dt.year)},
    }), "\n")

    curve = portfolio(t, bars, r.cost_pct)
    lo, hi = curve.index[0], curve.index[-1]
    rows = {f"pet, {SLOTS} slots": {**curve_stats(curve.equity), "avg slots used": curve.held.mean()}}
    for name, ticker in BENCHMARKS.items():
        px = yf.download(ticker, start=lo, end=hi + pd.Timedelta(days=1), auto_adjust=True,
                         progress=False)["Close"].squeeze().dropna()
        rows[name] = curve_stats(px)
    print(f"PORTFOLIO vs HOLDING A FUND, {lo.date()} to {hi.date()} (before tax)")
    print(table(rows), "\n")

    if args.symbol:
        s = t[t.symbol == args.symbol.upper()]
        cols = ["signal_date", "entry_date", "floor", "ceiling", "entry", "target", "stop",
                "exit_date", "exit", "reason", "ret", "days"]
        print(f"TRADES IN {args.symbol.upper()}")
        s = s[cols].assign(**{c: s[c].dt.date for c in ("signal_date", "entry_date", "exit_date")}).round(1)
        print(s.to_string(index=False) if len(s) else "none")


if __name__ == "__main__":
    main()
