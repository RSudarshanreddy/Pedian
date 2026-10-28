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

Deployed as the `form` Cloud Run service, called at 08:45 IST on weekdays (since
2026-10-28; it was 10:00 and 14:00), so the list arrives before the 9:00-9:08
pre-open order window: READY names bought at the opening price did 0.26-0.40 a
trade better than at any later time (ICICI 5-minute bars, all four halves). It
sends the list to the same Telegram chat as the momentum scanner.
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
GAUGE_MAX_AGE_DAYS = 4               # an older stored gauge counts as unavailable (a weekend is 3)
FIRE_COOLDOWN_SESSIONS = 10          # the test's no-re-entry window, in run days
BQ_TABLE = f"{PROJECT}.data_options.form_signals"
BQ_SCHEMA = (("run_date", "DATE"), ("run_timestamp", "TIMESTAMP"), ("prices_to", "DATE"), ("status", "STRING"),
             ("position", "INTEGER"), ("symbol", "STRING"), ("price", "FLOAT"), ("leg_pct", "FLOAT"),
             ("leg_days", "INTEGER"), ("up_days_pct", "FLOAT"), ("pullback_pct", "FLOAT"), ("target", "FLOAT"),
             ("exit_below", "FLOAT"), ("momentum_rank", "INTEGER"), ("day_change_pct", "FLOAT"),
             ("slot", "STRING"), ("source", "STRING"), ("day1", "BOOL"), ("gauge_pct", "FLOAT"))


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


def shortlist(t: pd.DataFrame, new: set[str] | None = None, n: int = LIST_SIZE) -> pd.DataFrame:
    """READY names: NEW entries first, never cut off, then the rest up to n; best leg
    (size x persistence) first within each. `new` is None when day-1 is unknown."""
    if t.empty:
        return t
    ready = t[t.candidate.eq(True)].sort_values("score", ascending=False).assign(status="READY")
    ready["day1"] = None if new is None else ready.index.isin(list(new))
    fresh = ready[ready.day1.eq(True)]
    rest = ready[~ready.day1.eq(True)]
    return pd.concat([fresh, rest.head(max(0, n - len(fresh)))])


def day1_symbols(symbols: list[str], today: dt.date) -> set[str]:
    """Which READY names are NEW entries by the test's own rule: the first day a
    stock is READY, and not within FIRE_COOLDOWN_SESSIONS of its last entry,
    replayed over the run days in form_signals (outside review, 2026-10-28: the
    message used to say BUY for names on their fifth day, which no test measured).
    Raises if the history cannot be read."""
    from google.cloud import bigquery
    rows = list(bigquery.Client(project=PROJECT).query(f"""
        WITH cal AS (SELECT DISTINCT run_date AS d FROM `{BQ_TABLE}`
                     WHERE run_date < @today ORDER BY d DESC LIMIT 40)
        SELECT 'cal' AS k, '' AS symbol, d FROM cal
        UNION DISTINCT
        SELECT 'on', symbol, run_date FROM `{BQ_TABLE}`
          WHERE run_date IN (SELECT d FROM cal) AND status = 'READY'""",
        job_config=bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("today", "DATE", today)])).result())
    days = sorted(r.d for r in rows if r.k == "cal") + [today]
    seen = {(r.symbol, r.d) for r in rows if r.k == "on"}
    new: set[str] = set()
    for sym in symbols:
        prev, last = False, -10 ** 6
        for i, d in enumerate(days):
            on = d == today or (sym, d) in seen
            if on and not prev and i - last > FIRE_COOLDOWN_SESSIONS:
                last = i
            prev = on
        if last == len(days) - 1:
            new.add(sym)
    return new


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
    too short for a 50-day average, so it reads the value momentum.py stores. At 08:45
    that is the previous day's run; the message says which day."""
    from google.cloud import bigquery
    rows = list(bigquery.Client(project=PROJECT).query(f"""
        SELECT Market_Breadth_Pct AS pct, Run_Date AS day FROM `{PROJECT}.data_options.momentum`
        WHERE Market_Breadth_Pct IS NOT NULL ORDER BY Run_Timestamp DESC LIMIT 1""").result())
    return (float(rows[0].pct), rows[0].day) if rows else None


def expected_session() -> dt.date | None:
    """The newest session the market has certainly traded before today: the bar date of
    momentum.py's latest earlier run (it runs at 10:00 on the day's own bar). If form's
    prices end before this, they are a session or more behind -- see run()."""
    from google.cloud import bigquery
    today = dt.datetime.now(IST).date()
    rows = list(bigquery.Client(project=PROJECT).query(f"""
        SELECT MAX(Bar_Date) AS d FROM `{PROJECT}.data_options.momentum` WHERE Run_Date < @today""",
        job_config=bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("today", "DATE", today)])).result())
    return rows[0].d if rows and rows[0].d else None


def sell_by(prices_to: dt.date) -> dt.date:
    """HOLD_SESSIONS weekdays after the last close used. NSE holidays are not known here,
    so around a holiday this lands a session early, never late."""
    return np.busday_offset(np.datetime64(prices_to, "D"), HOLD_SESSIONS, roll="forward").astype(dt.date)


def names_only(s: pd.DataFrame, gauge: tuple[float, dt.date] | None, prices_to: dt.date, source: str,
               expected: dt.date | None = None) -> str:
    """The Telegram message, in the format shared with momentum.py and swings.py
    (the owner's, 2026-10-28): the gauge, then Stock | Score | Action. Score is
    form's own -- leg size x share of up days, x100 -- the order READY is ranked
    in; it is not on the other two lists' scale. On a BUY day the action carries
    the plan: GTT at the peak, exit on a close below halfway, sell-by date.

    `expected` is the newest session already traded (expected_session). Prices that
    end before it are late -- NSE refused the cloud and Yahoo lagged -- and a READY
    judged on them is a different list, so nothing is marked BUY."""
    lines = [f"FORM  {dt.datetime.now(IST):%d %b %H:%M} IST -- prices close of {prices_to:%d %b}"
             + (" (Yahoo)" if source == "yahoo" else "")]
    late = expected is not None and prices_to < expected
    if late:
        lines.append(f"Prices are late: {expected:%d %b} has traded but is missing. WATCH ONLY today.")
    age = (dt.datetime.now(IST).date() - gauge[1]).days if gauge is not None else None
    if gauge is None:
        # Conservative: the gauge comes from another service's table, so "unknown" is
        # a real state here (e.g. before momentum.py's first run that stores it).
        buy = False
        lines.append("Market gauge unavailable -- WATCH ONLY until it is.")
    elif age > GAUGE_MAX_AGE_DAYS:
        # Stale: momentum.py has not stored a reading for days (outside review,
        # 2026-10-28). An old reading must not keep saying BUY.
        buy = False
        lines.append(f"Market gauge is from {gauge[1]:%d %b} ({age} days old) -- WATCH ONLY until it updates.")
    else:
        pct, day = gauge
        buy = pct >= GAUGE_MIN and not late
        lines.append(f"Market ({day:%d %b}): {pct:.0f}% of stocks above their 50-day average -- "
                     + ("WITH movers." if pct >= GAUGE_MIN else "AGAINST movers.\nNo new buys. List is WATCH ONLY."))
    lines.append("")
    if not len(s):
        lines.append("No names today.")
        return "\n".join(lines)
    lines.append("Stock | Score | Action")
    for sym, r in s.iterrows():
        new, unknown = r.get("day1") is True or r.get("day1") == True, r.get("day1") is None  # noqa: E712
        if not buy:
            action = "WATCH, new today" if new else "WATCH"
        elif unknown:
            action = "WATCH (day-1 history unavailable)"
        elif new:
            action = (f"BUY at the open, GTT {r.peak:,.0f}, exit below {r.halfway:,.0f}, "
                      f"sell by {sell_by(prices_to):%d %b}")
        else:
            action = "WATCH, not new"
        lines.append(f"{sym} | {float(r.score) * 100:.0f} | {action}")
    n_new = int(s["day1"].eq(True).sum()) if "day1" in s else 0
    lines.append("")
    lines.append(f"{len(s)} READY (healthy 4-7% pullback), {n_new} new today. "
                 f"Score orders the list; it is not a forecast.")
    if buy:
        lines.append("Only NEW names are tested buys -- the test bought on a stock's first READY day.")
        # "at the open": ICICI 5-minute bars, 2026-10-28 -- form names bought at
        # 09:20 or any later time did 0.26-0.40 a trade worse than at the opening
        # (pre-open) price, on all four halves. Hence the 08:45 run.
        # The gap rule (outside review, 2026-10-28): a stock that opens at or above
        # its peak has no upside left (the GTT would fire at once), and one that
        # opens at or below halfway has already broken its setup. Rare -- 3 of 250
        # gauge-ON trades -- so it changes the average by +0.02, but it stops a
        # pointless or broken trade at no cost.
        lines.append("Buy at the open: place orders 9:00-9:08 (pre-open). Skip a stock whose "
                     "pre-open price is above its GTT or below its exit line.")
        lines.append("Exit on a close below the line, or sell on the date. Same amount in each.")
    return "\n".join(lines)


def record(s: pd.DataFrame, prices_to: dt.date, source: str, gauge_pct: float | None = None) -> None:
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
             "pullback_pct": float(r.pullback) * 100, "target": float(r.peak), "exit_below": float(r.halfway),
             "day1": (None if r.get("day1") is None else bool(r.get("day1"))), "gauge_pct": gauge_pct}
            for i, (sym, r) in enumerate(s.iterrows(), 1)]
    client.query(f"DELETE FROM `{BQ_TABLE}` WHERE run_date = @d AND slot = @slot", job_config=bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("d", "DATE", now.date()),
                          bigquery.ScalarQueryParameter("slot", "STRING", slot)])).result()
    if rows:
        client.load_table_from_json(rows, BQ_TABLE, job_config=bigquery.LoadJobConfig(
            schema=schema, write_disposition="WRITE_APPEND",
            schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION])).result()
    print(f"recorded {len(rows)} rows in {BQ_TABLE} for {now.date()} {slot}")


def run() -> dict:
    """The day's list: details for the log, the names-only message, and what to record."""
    bars, source = prices()
    t = scan(bars)
    prices_to = last_session(bars).date()
    print(f"prices from {source} to {prices_to}; {len(t)} liquid stocks priced Rs {MIN_PRICE:.0f}+; "
          f"form: {t.form.value_counts().to_dict()}")
    ready = list(t.index[t.candidate.eq(True)]) if not t.empty else []
    try:
        new = day1_symbols(ready, dt.datetime.now(IST).date())
        print(f"day 1: {len(new)} of {len(ready)} READY names are new entries today")
    except Exception as exc:          # unknown day-1 -> no BUY marks, never a guess
        print(f"day-1 history unavailable: {exc}")
        new = None
    s = shortlist(t, new)
    try:
        gauge = market_gauge()
    except Exception as exc:          # the list still goes out if BigQuery is unreachable
        print(f"market gauge unavailable: {exc}")
        gauge = None
    print(f"market gauge: {gauge}")
    try:
        expected = expected_session()
    except Exception as exc:
        print(f"expected session unavailable: {exc}")
        expected = None
    if expected is not None and prices_to < expected:
        print(f"PRICES LATE: to {prices_to}, but {expected} has traded -- no BUY marks")
    return {"details": digest(s, prices_to), "names": names_only(s, gauge, prices_to, source, expected),
            "shortlist": s,
            "prices_to": prices_to, "source": source, "gauge_pct": gauge[0] if gauge else None}


def main(request: Any = None):
    """Cloud Run entry point (functions-framework) and command line. Only scheduled runs
    record to BigQuery; a request body of {"no_telegram": true} skips the message (for tests)."""
    out = run()
    print(out["details"])
    if request is None:
        return None
    body = request.get_json(silent=True) or {}
    try:
        record(out["shortlist"], out["prices_to"], out["source"], out["gauge_pct"])
    except Exception as exc:          # recording must never stop the message
        print(f"recording failed: {exc}")
    if not body.get("no_telegram"):
        send_telegram(out["names"])
    return out["names"], 200


if __name__ == "__main__":
    main()
