"""Movers in form, entered on a healthy pullback. The rules are in README.md; change them there first.

Scans every NSE stock using only its last 25 sessions:
  find the active leg -> judge its form -> keep the persistent movers
  -> READY: those in a healthy 4-7% pullback (up to 10 names, best leg first).

SINCE 2026-10-28 (README, "October 28"): READY only, behind the market gauge, with
its exit plan in the message -- sell at the leg's peak (GTT), sell on a close
below halfway, and sell by 10 sessions at the latest. Tested with rules fixed
first, Dec-Sep, entry the day a stock first turns READY:
    gauge ON   260 trades  +1.21% a trade  67% won  5.6 sessions held  every split positive
    gauge OFF   94 trades  -0.73%          47% won
Live 9-28 Oct, all on gauge-OFF days, READY lost 3.7% a trade on this plan.
The gauge is the share of every stock in momentum.py's price/liquidity range above
its 50-day average, read from the latest momentum run; below 50% READY is WATCH ONLY.
WATCH (now momentum.py's job) and MOMENTUM DIPS (built on the retired Expected_Move
top 20; 8% of its live picks made money) were removed. It shares no trades with
swings.py, which buys deeper (12%+) dips.

    python form.py        # downloads ~3 months for every NSE stock (a few minutes)

Prices come from NSE's own end-of-day files (data.nse_bars), out the same evening.
Yahoo was a day or two late -- on 9 Oct it still lacked 8 Oct, which changed 7 of
the 10 names -- so it is only the fallback, and the message then says so.

Deployed as the `form` Cloud Run service, called at 10:00 and 14:00 IST on
weekdays; it sends the list to the same Telegram chat as the momentum scanner.
A bar for today is never used before the close -- the list is always judged on
complete sessions, as in the test -- and the message names the close it is on.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import urllib.request
from typing import Any

import numpy as np
import pandas as pd

import data

WINDOW = 25
SESSIONS = WINDOW + 5                 # fetched from NSE, so a stock missing a few days still fills the window
MIN_PRICE = 100.0
MIN_TRADED_CR = 10.0
MAX_SHOWN = 30
LIST_SIZE = 10
PROJECT = "sudarshan-442212"
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
CLOSE_DONE = dt.time(15, 45)         # after this a bar for today is a complete session
GAUGE_MIN = 50.0                     # market gauge: below this READY is WATCH ONLY
HOLD_SESSIONS = 10                   # the plan's latest exit
BQ_TABLE = f"{PROJECT}.data_options.form_signals"
BQ_SCHEMA = (("run_date", "DATE"), ("run_timestamp", "TIMESTAMP"), ("prices_to", "DATE"), ("status", "STRING"),
             ("position", "INTEGER"), ("symbol", "STRING"), ("price", "FLOAT"), ("leg_pct", "FLOAT"),
             ("leg_days", "INTEGER"), ("up_days_pct", "FLOAT"), ("pullback_pct", "FLOAT"), ("target", "FLOAT"),
             ("exit_below", "FLOAT"), ("momentum_rank", "INTEGER"), ("day_change_pct", "FLOAT"),
             ("slot", "STRING"), ("source", "STRING"))


def analyse(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, v: np.ndarray) -> dict:
    """The leg, its form and the pullback since its peak, from one WINDOW of sessions (oldest first)."""
    p = int(np.argmax(c))
    if p == 0:
        return {"form": "noise"}
    s = int(np.argmin(c[:p + 1]))
    size = c[p] / c[s] - 1
    length = p - s
    m = {"start": c[s], "peak": c[p], "halfway": c[s] + (c[p] - c[s]) / 2, "size": size, "length": length}
    if size <= 0 or length == 0:
        return {**m, "form": "noise"}

    r = c[s + 1:p + 1] / c[s:p] - 1
    total = np.log(c[p] / c[s])
    m["persistence"] = float(np.mean(r > 0))
    m["biggest_day"] = float(np.max(np.log1p(r)) / total)
    m["gap_share"] = float(np.sum(np.maximum(0.0, np.log(o[s + 1:p + 1] / c[s:p]))) / total)
    leg = c[s:p + 1]
    m["inner_dip"] = float(np.min(leg / np.maximum.accumulate(leg)) - 1)
    m["volume"] = float(np.mean(v[s + 1:p + 1]) / np.mean(v)) if np.mean(v) > 0 else np.nan
    mid = s + length // 2
    m["acceleration"] = float((c[p] / c[mid] - 1) - (c[mid] / c[s] - 1)) if length >= 2 else np.nan

    pb_days = len(c) - 1 - p
    m["pullback"] = c[-1] / c[p] - 1
    m["pullback_days"] = pb_days
    if pb_days:
        m["pullback_worst_day"] = float(np.min(c[p + 1:] / c[p:-1] - 1))
        leg_vol = np.mean(v[s + 1:p + 1])
        m["pullback_volume"] = float(np.mean(v[p + 1:]) / leg_vol) if leg_vol > 0 else np.nan
        m["below_halfway"] = bool(np.any(c[p + 1:] <= m["halfway"]))
    else:
        m.update(pullback_worst_day=0.0, pullback_volume=np.nan, below_halfway=False)

    if size < 0.10:
        form = "noise"
    elif length < 5 or m["biggest_day"] >= 0.5 or m["gap_share"] >= 0.5:
        form = "burst"
    elif m["pullback"] < -0.12 or pb_days > 10 or m["below_halfway"]:
        form = "exhausted"
    elif size < 0.15:
        form = "emerging"
    elif (m["persistence"] >= 0.55 and m["biggest_day"] < 0.4 and m["gap_share"] < 0.4
          and m["inner_dip"] > -0.10 and m["volume"] >= 1.0):
        form = "persistent"
    else:
        form = "burst"
    m["form"] = form
    m["healthy_pullback"] = healthy_pullback(m)
    m["candidate"] = form == "persistent" and m["healthy_pullback"]
    m["score"] = size * m["persistence"]
    return m


def healthy_pullback(m: dict) -> bool:
    return (-0.07 <= m["pullback"] <= -0.04 and 1 <= m["pullback_days"] <= 5
            and m["pullback_worst_day"] > -0.05 and m["pullback_volume"] < 1.0 and not m["below_halfway"])


def in_universe(c: np.ndarray, v: np.ndarray) -> bool:
    return c[-1] >= MIN_PRICE and np.mean(c[-20:] * v[-20:]) / 1e7 >= MIN_TRADED_CR


def scan(bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Every stock's form as of its latest bar."""
    latest = max(df.index[-1] for df in bars.values())
    rows = {}
    for sym, df in bars.items():
        if len(df) < WINDOW or df.index[-1] < latest - pd.Timedelta(days=4):
            continue
        w = df.iloc[-WINDOW:]
        o, h, l, c, v = (w[k].to_numpy(float) for k in ("Open", "High", "Low", "Close", "Volume"))
        if not in_universe(c, v):
            continue
        rows[sym] = {"date": w.index[-1].date(), "price": c[-1], **analyse(o, h, l, c, v)}
    return pd.DataFrame(rows).T


def shortlist(t: pd.DataFrame, n: int = LIST_SIZE) -> pd.DataFrame:
    """READY names, best leg (size x persistence) first."""
    if t.empty:
        return t
    return t[t.candidate.eq(True)].sort_values("score", ascending=False).assign(status="READY").head(n)


def digest(s: pd.DataFrame, prices_to: dt.date) -> str:
    """Plain text: Telegram's Markdown parser chokes on stray underscores in data."""
    lines = [f"form.py -- {dt.datetime.now(IST):%d %b %Y} (prices to {prices_to:%d %b})", ""]
    for status, title in (("READY", "READY -- in a healthy 4-7% pullback"),):
        rows = s[s.status == status]
        if not len(rows):
            continue
        lines.append(f"{title}:")
        for sym, r in rows.iterrows():
            head = f"{sym} {r.price:,.1f} | leg +{r['size'] * 100:.0f}% in {int(r.length)}d"
            if status == "READY":
                lines.append(f"{head} | {r.pullback * 100:+.1f}% from peak | target {r.peak:,.1f} | "
                             f"exit below {r.halfway:,.1f}")
            else:
                lines.append(f"{head} | buy zone {r.peak * 0.93:,.1f}-{r.peak * 0.96:,.1f}")
        lines.append("")
    if not len(s):
        lines.append("No persistent movers today.")
    lines.append("Check live prices in Kite before acting.")
    return "\n".join(lines)


def universe() -> list[str]:
    """NSE's equity list; NSE often refuses cloud servers, so fall back to the
    copy the momentum scanner keeps in BigQuery."""
    try:
        return data.load_universe()
    except Exception:
        from google.cloud import bigquery
        rows = bigquery.Client(project=PROJECT).query(
            f"SELECT Ticker FROM `{PROJECT}.data_options.nse_ticker_cache`").result()
        return sorted({r.Ticker.removesuffix(".NS") for r in rows})


def complete_sessions_only(bars: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    now = dt.datetime.now(IST)
    if now.time() >= CLOSE_DONE:
        return bars
    return {s: df[df.index.date < now.date()] for s, df in bars.items()}


def prices() -> tuple[dict[str, pd.DataFrame], str]:
    """Every stock's recent sessions, from NSE's end-of-day files; Yahoo only if NSE can't be reached."""
    symbols = universe()
    try:
        bars = data.nse_bars(symbols, SESSIONS, dt.datetime.now(IST).date())
        source = "nse"
    except Exception as exc:
        print(f"NSE files unavailable ({exc}); falling back to Yahoo")
        bars = {}
    if len({d for df in bars.values() for d in df.index}) < WINDOW:
        bars, source = data.download(symbols, period="3mo", min_rows=WINDOW), "yahoo"
    bars = complete_sessions_only(bars)
    return {s: df for s, df in bars.items() if len(df) >= WINDOW}, source


def send_telegram(text: str) -> None:
    token, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("Telegram skipped: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set.")
        return
    req = urllib.request.Request(f"https://api.telegram.org/bot{token}/sendMessage",
                                 data=json.dumps({"chat_id": chat, "text": text}).encode(),
                                 headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=15).read()


def last_session(bars: dict[str, pd.DataFrame]) -> pd.Timestamp:
    """The latest session MOST stocks have. Yahoo publishes some stocks a day or two ahead
    of the rest, so the newest date of any single stock would make the rest look stale."""
    return pd.Series([df.index[-1] for df in bars.values()]).mode().iloc[0]


def market_gauge() -> tuple[float, dt.date] | None:
    """The market gauge from the latest momentum run: the share of every stock in its
    price/liquidity range above its own 50-day average. form's own bars are 30 sessions,
    too short for a 50-day average, so it reads the value momentum.py stores. At 10:00
    that is usually the previous day's run; the message says which day."""
    from google.cloud import bigquery
    rows = list(bigquery.Client(project=PROJECT).query(f"""
        SELECT Market_Breadth_Pct AS pct, Run_Date AS day FROM `{PROJECT}.data_options.momentum`
        WHERE Market_Breadth_Pct IS NOT NULL ORDER BY Run_Timestamp DESC LIMIT 1""").result())
    return (float(rows[0].pct), rows[0].day) if rows else None


def sell_by(prices_to: dt.date) -> dt.date:
    """HOLD_SESSIONS weekdays after the last close used. NSE holidays are not known here,
    so around a holiday this lands a session early, never late."""
    return np.busday_offset(np.datetime64(prices_to, "D"), HOLD_SESSIONS, roll="forward").astype(dt.date)


def names_only(s: pd.DataFrame, gauge: tuple[float, dt.date] | None, prices_to: dt.date, source: str) -> str:
    """The Telegram message: the gauge, then each READY name with its plan."""
    lines = [f"form {dt.datetime.now(IST):%d %b}, on {prices_to:%d %b} close" + (" (Yahoo)" if source == "yahoo" else "")]
    if gauge is None:
        # Conservative: the gauge comes from another service's table, so "unknown" is
        # a real state here (e.g. before momentum.py's first run that stores it).
        buy = False
        lines.append("Market gauge unavailable -- WATCH ONLY until it is.")
    else:
        pct, day = gauge
        buy = pct >= GAUGE_MIN
        lines.append(f"Market ({day:%d %b}): {pct:.0f}% of stocks above their 50-day average -- "
                     + ("WITH movers." if buy else "AGAINST movers.\nNo new buys. List is WATCH ONLY."))
    lines.append("")
    if not len(s):
        lines.append("READY: none today.")
        return "\n".join(lines)
    if buy:
        lines.append(f"READY -- buy. GTT sell at the peak; sell on a close below the exit; "
                     f"sell by {sell_by(prices_to):%d %b} at the latest")
    else:
        lines.append("READY (watch only) -- in a healthy 4-7% pullback")
    width = max(len(sym) for sym in s.index)
    for sym, r in s.iterrows():
        lines.append(f"  {sym:<{width}}  {r.pullback * 100:+4.0f}%  peak {r.peak:,.0f}  exit below {r.halfway:,.0f}")
    return "\n".join(lines)


def record(s: pd.DataFrame, prices_to: dt.date, source: str) -> None:
    """Keep what was sent, for the end-of-October and December reviews. Re-running in the same
    slot (morning, afternoon, or evening after the close) replaces that slot's rows. A load job, not
    streaming, so the DELETE always works."""
    from google.cloud import bigquery
    client = bigquery.Client(project=PROJECT)
    schema = [bigquery.SchemaField(n, t) for n, t in BQ_SCHEMA]
    client.create_table(bigquery.Table(BQ_TABLE, schema=schema), exists_ok=True)
    now = dt.datetime.now(IST)
    slot = "morning" if now.hour < 12 else "afternoon" if now.time() < CLOSE_DONE else "evening"
    base = {"run_date": now.date().isoformat(), "run_timestamp": now.isoformat(), "prices_to": prices_to.isoformat(),
            "slot": slot, "source": source}
    rows = [{**base, "status": r.status, "position": i, "symbol": sym, "price": float(r.price),
             "leg_pct": float(r["size"]) * 100, "leg_days": int(r.length), "up_days_pct": float(r.persistence) * 100,
             "pullback_pct": float(r.pullback) * 100, "target": float(r.peak), "exit_below": float(r.halfway)}
            for i, (sym, r) in enumerate(s.iterrows(), 1)]
    client.query(f"DELETE FROM `{BQ_TABLE}` WHERE run_date = @d AND slot = @slot", job_config=bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("d", "DATE", now.date()),
                          bigquery.ScalarQueryParameter("slot", "STRING", slot)])).result()
    if rows:
        client.load_table_from_json(rows, BQ_TABLE, job_config=bigquery.LoadJobConfig(
            schema=schema, write_disposition="WRITE_APPEND")).result()
    print(f"recorded {len(rows)} rows in {BQ_TABLE} for {now.date()} {slot}")


def run() -> dict:
    """The day's list: details for the log, the names-only message, and what to record."""
    bars, source = prices()
    t = scan(bars)
    prices_to = last_session(bars).date()
    print(f"prices from {source} to {prices_to}; {len(t)} liquid stocks priced Rs {MIN_PRICE:.0f}+; "
          f"form: {t.form.value_counts().to_dict()}")
    s = shortlist(t)
    try:
        gauge = market_gauge()
    except Exception as exc:          # the list still goes out if BigQuery is unreachable
        print(f"market gauge unavailable: {exc}")
        gauge = None
    print(f"market gauge: {gauge}")
    return {"details": digest(s, prices_to), "names": names_only(s, gauge, prices_to, source), "shortlist": s,
            "prices_to": prices_to, "source": source}


def main(request: Any = None):
    """Cloud Run entry point (functions-framework) and command line. Only scheduled runs
    record to BigQuery; a request body of {"no_telegram": true} skips the message (for tests)."""
    out = run()
    print(out["details"])
    if request is None:
        return None
    body = request.get_json(silent=True) or {}
    try:
        record(out["shortlist"], out["prices_to"], out["source"])
    except Exception as exc:          # recording must never stop the message
        print(f"recording failed: {exc}")
    if not body.get("no_telegram"):
        send_telegram(out["names"])
    return out["names"], 200


if __name__ == "__main__":
    main()
