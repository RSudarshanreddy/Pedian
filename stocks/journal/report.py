"""Trading feedback from a Kite tradebook: where the money came from, and where it leaked.

    python report.py TRADEBOOK.csv [MORE.csv ...] --holdings HOLDINGS.csv

TRADEBOOK.csv is Kite Console > Reports > Tradebook (Equity). Several files are
merged and de-duplicated, so overlapping date ranges are fine.
HOLDINGS.csv has the columns symbol,isin,quantity,average_price,last_price.

Sells are matched to buys first-in-first-out per ISIN, as Zerodha does.
Shares bought before the tradebook starts have no known cost: selling them is
listed on its own and never guessed at. Charges are ESTIMATED (delivery STT,
stamp duty, exchange fees, GST, DP charge); the contract notes are exact.

Keep the CSVs out of git -- they are personal financial records.
"""

from __future__ import annotations

import argparse
from collections import deque

import numpy as np
import pandas as pd

BUY_CHARGES = 0.0012    # STT 0.1% + stamp 0.015% + exchange/SEBI/GST ~0.005%
SELL_CHARGES = 0.00105  # STT 0.1% + exchange/SEBI/GST ~0.005%
DP_PER_SELL = 15.34     # per stock, per day sold (from the ledger)
SPLIT_SUSPECT = 0.5     # a rebuy 50%+ away from the sell price is probably a split or demerger


def rupees(x: float) -> str:
    return "" if pd.isna(x) else f"{x:+,.0f}"


def load_trades(paths: list[str]) -> pd.DataFrame:
    t = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    t = t.drop_duplicates(subset=["exchange", "trade_id", "order_id"])
    t["time"] = pd.to_datetime(t.order_execution_time)
    t["date"] = pd.to_datetime(t.trade_date)
    return t.sort_values(["time", "trade_type"]).reset_index(drop=True)


def match_fifo(t: pd.DataFrame, pre_qty: dict[str, float]):
    """Returns (matches, open_lots). A match's buy_price is NaN for pre-tradebook shares."""
    lots = {isin: deque([[q, np.nan, pd.NaT]]) for isin, q in pre_qty.items() if q > 0}
    matches = []
    for r in t.itertuples():
        book = lots.setdefault(r.isin, deque())
        if r.trade_type == "buy":
            book.append([r.quantity, r.price, r.date])
            continue
        left = r.quantity
        while left > 1e-9 and book:
            lot = book[0]
            q = min(left, lot[0])
            matches.append(dict(isin=r.isin, sell_date=r.date, qty=q, buy_price=lot[1],
                                buy_date=lot[2], sell_price=r.price))
            lot[0] -= q
            left -= q
            if lot[0] <= 1e-9:
                book.popleft()
        if left > 1e-9:
            raise ValueError(f"{r.symbol}: sold {left} more than the tradebook and holdings explain")
    return pd.DataFrame(matches), lots


def closed_trades(m: pd.DataFrame, names: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per decision to sell (stock + day), split into known-cost and pre-tradebook parts."""
    m = m.assign(cost=m.qty * m.buy_price, proceeds=m.qty * m.sell_price,
                 held=(m.sell_date - m.buy_date).dt.days * m.qty)
    known, pre = m[m.buy_price.notna()], m[m.buy_price.isna()]
    k = known.groupby(["isin", "sell_date"]).agg(qty=("qty", "sum"), cost=("cost", "sum"),
                                                 proceeds=("proceeds", "sum"), held=("held", "sum")).reset_index()
    k["stock"] = k["isin"].map(names)
    k["days"] = k.held / k.qty
    k["charges"] = k.cost * BUY_CHARGES + k.proceeds * SELL_CHARGES + DP_PER_SELL
    k["net"] = k.proceeds - k.cost - k.charges
    k["pct"] = k.net / k.cost * 100
    p = pre.groupby(["isin", "sell_date"]).agg(qty=("qty", "sum"), proceeds=("proceeds", "sum")).reset_index()
    p["stock"] = p["isin"].map(names)
    return k, p


def rebuys(t: pd.DataFrame, names: dict[str, str], now: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For every sell decision: did you buy the same stock back, and at what price?"""
    sells = (t[t.trade_type == "sell"].assign(value=lambda x: x.quantity * x.price)
             .groupby(["isin", "date"]).agg(qty=("quantity", "sum"), value=("value", "sum"),
                                            last=("time", "max")).reset_index())
    rows, skipped = [], []
    for s in sells.itertuples():
        sell_px = s.value / s.qty
        later = t[(t["isin"] == s.isin) & (t.trade_type == "buy") & (t.time > s.last)]
        left, paid, first = s.qty, 0.0, None
        for b in later.itertuples():
            if left <= 0:
                break
            if abs(b.price / sell_px - 1) >= SPLIT_SUSPECT:
                skipped.append(f"{names[s.isin]} sold {s.date.date()} at {sell_px:,.1f}, "
                               f"rebought {b.date.date()} at {b.price:,.1f}")
                left = 0
                break
            q = min(left, b.quantity)
            paid += q * (b.price - sell_px)
            left -= q
            first = first or b.date
        rebought = s.qty - left
        rows.append(dict(stock=names[s.isin], sold=s.date.date(), qty=s.qty, sell_price=sell_px,
                         rebought=rebought, days_to_rebuy=(first - s.date).days if first is not None else np.nan,
                         paid_extra=paid if rebought else np.nan,
                         not_rebought=left, now=now.get(s.isin, np.nan),
                         missed=left * (now.get(s.isin, np.nan) - sell_px) if left else np.nan))
    return pd.DataFrame(rows), pd.Series(skipped, dtype=str)


def bucket(x: pd.Series, edges: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(x, edges, labels=labels, right=False)


def summarise(df: pd.DataFrame, by: str, pnl: str) -> pd.DataFrame:
    g = df.groupby(by, observed=True)
    return pd.DataFrame({"trades": g.size(), "rupees": g[pnl].sum().map(rupees),
                         "winners %": g[pnl].apply(lambda s: (s > 0).mean() * 100).round(0)})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tradebooks", nargs="+")
    ap.add_argument("--holdings", required=True)
    ap.add_argument("--no-fetch", action="store_true", help="don't look up today's price for stocks you no longer hold")
    args = ap.parse_args()

    t = load_trades(args.tradebooks)
    h = pd.read_csv(args.holdings)
    names = {**t.groupby("isin").symbol.last().to_dict(), **h.set_index("isin").symbol.to_dict()}
    net_bought = t.assign(s=np.where(t.trade_type == "buy", t.quantity, -t.quantity)).groupby("isin").s.sum()
    held = h.set_index("isin").quantity
    pre_qty = held.reindex(net_bought.index.union(held.index), fill_value=0).sub(net_bought, fill_value=0)
    if (pre_qty < -1e-9).any():
        raise ValueError(f"holdings are smaller than the tradebook allows: {pre_qty[pre_qty < 0].to_dict()}")

    m, lots = match_fifo(t, pre_qty.to_dict())
    k, p = closed_trades(m, names)

    now = h.set_index("isin").last_price.to_dict()
    gone = sorted(set(t["isin"]) - set(now))
    if not args.no_fetch and gone:
        import yfinance as yf
        for isin in gone:
            for suffix in (".NS", ".BO"):
                px = yf.download(names[isin] + suffix, period="5d", progress=False, auto_adjust=False)
                if len(px):
                    now[isin] = float(px["Close"].squeeze().dropna().iloc[-1])
                    break
    rb, skipped = rebuys(t, names, now)

    first, last = t.date.min().date(), t.date.max().date()
    print(f"TRADEBOOK {first} to {last}: {len(t)} fills, {len(k)} sells with a known cost, "
          f"{len(p)} sells of shares bought before {first}\n")

    # 1. the money
    h["cost"] = h.quantity * h.average_price
    h["value"] = h.quantity * h.last_price
    h["pnl"] = h.value - h.cost
    buys = t[t.trade_type == "buy"].eval("quantity * price").sum()
    sells = t[t.trade_type == "sell"].eval("quantity * price").sum()
    charges = buys * BUY_CHARGES + sells * SELL_CHARGES + DP_PER_SELL * len(k.drop_duplicates(["isin", "sell_date"]))
    print("1. THE MONEY")
    print(f"   closed trades (known cost), after est. charges : {rupees(k.net.sum())}  "
          f"({(k.net > 0).sum()} of {len(k)} won)")
    print(f"   sold from before {first} (cost unknown)       : {p.proceeds.sum():,.0f} received")
    print(f"   open positions, not yet sold                    : {rupees(h.pnl.sum())}  on {h.cost.sum():,.0f} invested")
    print(f"   estimated charges paid in the period            : {-charges:,.0f}  "
          f"({charges / (buys + sells) * 100:.2f}% of {buys + sells:,.0f} traded)\n")

    # 2. closed trades
    print("2. CLOSED TRADES, best and worst (after estimated charges)")
    cols = ["stock", "sell_date", "qty", "cost", "net", "pct", "days"]
    view = k.sort_values("net")[cols].assign(sell_date=lambda x: x.sell_date.dt.date,
                                             cost=lambda x: x.cost.map("{:,.0f}".format),
                                             net=lambda x: x.net.map(rupees), pct=lambda x: x.pct.round(1),
                                             days=lambda x: x.days.round(0))
    print(pd.concat([view.tail(5).iloc[::-1], view.head(5)]).to_string(index=False), "\n")

    # 3. what the feedback says
    k["size"] = bucket(k.cost, [0, 25e3, 50e3, 1e5, np.inf], ["< 25k", "25-50k", "50k-1L", "> 1L"])
    h["size"] = bucket(h.cost, [0, 25e3, 50e3, 1e5, np.inf], ["< 25k", "25-50k", "50k-1L", "> 1L"])
    print("3a. BY POSITION SIZE (money put in)")
    print(pd.concat({"closed": summarise(k, "size", "net"), "open now": summarise(h, "size", "pnl")},
                    axis=1).to_string(), "\n")
    k["held"] = bucket(k.days, [0, 2, 6, 16, 31, np.inf], ["0-1 days", "2-5 days", "6-15 days", "16-30 days", "> 30 days"])
    print("3b. CLOSED TRADES BY DAYS HELD")
    print(summarise(k, "held", "net").to_string(), "\n")

    back = rb[rb.rebought > 0]
    print("3c. SOLD, THEN BOUGHT THE SAME STOCK BACK")
    print(f"   {len(back)} of {len(rb)} sells were followed by a rebuy; "
          f"median {back.days_to_rebuy.median():.0f} days later")
    print(f"   buying back cost {rupees(back.paid_extra.sum())} more than the sell price "
          f"(positive = you paid more to get back in)")
    worst = back.sort_values("paid_extra", ascending=False).head(5)
    print(worst[["stock", "sold", "sell_price", "rebought", "days_to_rebuy", "paid_extra"]]
          .assign(sell_price=lambda x: x.sell_price.round(1), paid_extra=lambda x: x.paid_extra.map(rupees))
          .to_string(index=False))
    out = rb[rb.not_rebought > 0].dropna(subset=["missed"])
    gap = out.missed.sum()
    print(f"\n   shares sold and never bought back are worth {abs(gap):,.0f} {'MORE' if gap > 0 else 'LESS'} "
          f"today than you sold them for ({len(out)} sells) -- "
          f"{'selling them cost you that upside' if gap > 0 else 'selling them avoided that fall'}")
    ends = out.sort_values("missed")
    print(pd.concat([ends.head(3), ends.tail(3)])[["stock", "sold", "not_rebought", "sell_price", "now", "missed"]]
          .assign(sell_price=lambda x: x.sell_price.round(1), now=lambda x: x.now.round(1),
                  missed=lambda x: x.missed.map(rupees)).to_string(index=False))
    for s in skipped:
        print(f"   skipped, probably a split or demerger: {s}")
    print()

    # 4. open positions
    oldest = {isin: min((lot[2] for lot in book if lot[0] > 1e-9), default=pd.NaT, key=lambda d: (pd.notna(d), d))
              for isin, book in lots.items()}
    h["since"] = h["isin"].map(lambda i: "before " + str(first) if pd.isna(oldest.get(i, pd.NaT))
                            else oldest[i].date().isoformat())
    h["share %"] = h.cost / h.cost.sum() * 100
    h["pct"] = h.pnl / h.cost * 100
    print("4. OPEN POSITIONS, largest first")
    print(h.sort_values("cost", ascending=False)[["symbol", "cost", "pnl", "pct", "share %", "since"]]
          .assign(cost=lambda x: x.cost.map("{:,.0f}".format), pnl=lambda x: x.pnl.map(rupees),
                  pct=lambda x: x.pct.round(1), **{"share %": lambda x: x["share %"].round(1)})
          .to_string(index=False))
    odd = h[h.pct.abs() >= 40]
    if len(odd):
        print(f"\n   check in Console for a split/bonus: {', '.join(odd.symbol)} "
              f"(moved {', '.join(f'{v:+.0f}%' for v in odd.pct)} from average price)")


if __name__ == "__main__":
    main()
