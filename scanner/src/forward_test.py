"""
FORWARD TEST -- does a signal from swings.py actually make money?

This is the piece ARCHITECTURE.md calls "the missing piece". Everything the
scanner measures describes STOCKS: "this one historically makes large
asymmetric moves". Nothing measured whether the SIGNAL works: "when the
scanner listed this on 2026-09-18, would buying it have paid".

Those are different questions, and only the second one decides whether any of
the thresholds are worth what they cost. Until it is answered, every gate in
swings.py is set on judgment.

WHAT THIS DOES
  Reads stored signals from data_options.swings, waits for each one's forward
  window to complete, then replays what actually happened with real prices and
  writes the outcome to data_options.signal_outcomes. One row per
  (Ticker, Bar_Date, Horizon_Days).

  It is idempotent -- already-evaluated signals are skipped -- so it is safe to
  run on a schedule and safe to re-run after a failure.

WHAT MAKES IT HONEST
  1. TWO ENTRY CONVENTIONS, both recorded. Entry_Close is the signal bar's
     close, which is what every backtest in this project assumed and is NOT
     reachable -- the signal does not exist until that bar has closed.
     Entry_Open is the NEXT session's open, which is the first price you could
     genuinely transact at. The gap between them is the cost of the assumption,
     and reporting both means nobody has to take my word for how big it is.
  2. Costs charged on every outcome at config.round_trip_cost_pct.
  3. Stop and target come from the STORED row, not recomputed, so the test
     grades the signal that was actually issued rather than a tidied-up version
     of it.
  4. The same data-integrity path as the scanner (get_ticker_frame drops
     Yahoo's fabricated zero-volume holiday bars), so outcomes are not measured
     against sessions that never happened.

WHAT IT DELIBERATELY DOES NOT DO
  Model an exit. Stop_Hit and Target_Hit are recorded as facts about the price
  path -- did it touch that level, and on which session -- not as a simulated
  trade. Combining them into a P&L requires deciding which fires first
  intraday, which daily bars cannot answer. Exits stay a human decision; this
  measures what the market offered.

USAGE
  python forward_test.py --backfill          # evaluate everything that is ready
  python forward_test.py --report            # analyse what has been evaluated
  python forward_test.py --backfill --report
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from google.api_core.exceptions import NotFound
from google.cloud import bigquery

import swings as S

logging.basicConfig(level=logging.WARNING)
LOGGER = logging.getLogger(__name__)

OUTCOME_TABLE_ID = "signal_outcomes"

# Horizons evaluated for every signal, in SESSIONS.
#
# 30 is the scanner's own move_horizon_days -- the window Expected_Move is
# measured over -- so it is the horizon the gate is actually a claim about.
# The shorter ones are here because a swing trade is not held for six weeks and
# because the shape of the curve matters: a signal that pays at 5 and gives it
# back by 30 is a different animal from one that keeps running.
HORIZONS = (5, 10, 15, 30)

OUTCOME_SCHEMA = [
    bigquery.SchemaField("Ticker", "STRING"),
    bigquery.SchemaField("Bar_Date", "DATE"),
    bigquery.SchemaField("Horizon_Days", "INT64"),
    bigquery.SchemaField("Evaluated_At", "TIMESTAMP"),
    # --- what the scanner said at the time ---
    bigquery.SchemaField("Action", "STRING"),
    bigquery.SchemaField("Setup_Type", "STRING"),
    bigquery.SchemaField("Score", "FLOAT64"),
    bigquery.SchemaField("Expected_Move", "FLOAT64"),
    bigquery.SchemaField("Avg_Volatility", "FLOAT64"),
    bigquery.SchemaField("Tail_Drawdown_Pct", "FLOAT64"),
    bigquery.SchemaField("Spike_Ratio", "FLOAT64"),
    bigquery.SchemaField("Max_Drawdown_1Y_Pct", "FLOAT64"),
    bigquery.SchemaField("Traded_Value_Cr", "FLOAT64"),
    bigquery.SchemaField("Risk_Pct", "FLOAT64"),
    bigquery.SchemaField("Signal_Price", "FLOAT64"),
    bigquery.SchemaField("Stop_Loss", "FLOAT64"),
    bigquery.SchemaField("Typical_Move_Price", "FLOAT64"),
    # --- what actually happened ---
    bigquery.SchemaField("Entry_Close", "FLOAT64"),
    bigquery.SchemaField("Entry_Open", "FLOAT64"),
    bigquery.SchemaField("Sessions_Available", "INT64"),
    bigquery.SchemaField("Return_Close_Pct", "FLOAT64"),
    bigquery.SchemaField("Return_Open_Pct", "FLOAT64"),
    bigquery.SchemaField("MFE_Pct", "FLOAT64"),
    bigquery.SchemaField("MAE_Pct", "FLOAT64"),
    bigquery.SchemaField("Stop_Hit", "BOOL"),
    bigquery.SchemaField("Sessions_To_Stop", "INT64"),
    bigquery.SchemaField("Target_Hit", "BOOL"),
    bigquery.SchemaField("Sessions_To_Target", "INT64"),
]


def load_pending_signals(client: bigquery.Client, project: str, dataset: str,
                         table: str, limit: int = 0) -> pd.DataFrame:
    """Signals that are not yet evaluated at the longest horizon.

    De-duplicated to one row per (Ticker, Bar_Date). The scanner runs several
    times a day and writes a row each time, but those are repeated looks at the
    SAME bar, not separate signals -- counting them separately would weight a
    ticker by how many times it happened to be scanned. The earliest run of the
    day is kept, because that is the first moment the signal was available.
    """
    outcome_ref = f"{project}.{dataset}.{OUTCOME_TABLE_ID}"
    try:
        client.get_table(outcome_ref)
        already = f"""
        LEFT JOIN (
          SELECT Ticker t, Bar_Date b FROM `{outcome_ref}`
          WHERE Horizon_Days = {max(HORIZONS)}
        ) done ON done.t = s.Ticker AND done.b = s.Bar_Date
        """
        where_done = "AND done.t IS NULL"
    except NotFound:
        already, where_done = "", ""

    # Columns are selected against the table's ACTUAL schema, not a fixed list.
    # Spike_Ratio and Max_Drawdown_1Y_Pct were added to swings.py after this
    # table already held a month of rows, and BigQuery adds a column only when
    # a load first supplies it -- so on any table written before that change
    # they simply do not exist and naming them is a hard query error. Anything
    # absent is selected as NULL instead, which is also the truthful value: the
    # scanner did not measure it when that signal was issued.
    existing = {f.name for f in client.get_table(f"{project}.{dataset}.{table}").schema}
    wanted = [
        ("Ticker", "Ticker"), ("Bar_Date", "Bar_Date"), ("Action", "Action"),
        ("Setup_Type", "Setup_Type"), ("Score", "Score"),
        ("Expected_Move", "Expected_Move"), ("Avg_Volatility", "Avg_Volatility"),
        ("Tail_Drawdown_Pct", "Tail_Drawdown_Pct"), ("Spike_Ratio", "Spike_Ratio"),
        ("Max_Drawdown_1Y_Pct", "Max_Drawdown_1Y_Pct"),
        ("Traded_Value_Cr", "Traded_Value_Cr"), ("Risk_Pct", "Risk_Pct"),
        ("Price", "Signal_Price"), ("Stop_Loss", "Stop_Loss"),
    ]
    select = ", ".join(
        f"{src} AS {alias}" if src in existing else f"CAST(NULL AS FLOAT64) AS {alias}"
        for src, alias in wanted
    )
    # The target column was renamed Target -> Typical_Move_Price partway through
    # this table's life, so neither name alone covers it: 478 rows carry the new
    # name and 1,135 the old one. Reading only the new name made Target_Hit come
    # back 0.0% at every horizon -- which looked like a finding ("the target is
    # never reached") and was actually just a null column. COALESCE covers both.
    target_parts = [c for c in ("Typical_Move_Price", "Target") if c in existing]
    select += (f", COALESCE({', '.join(target_parts)}) AS Typical_Move_Price"
               if target_parts else ", CAST(NULL AS FLOAT64) AS Typical_Move_Price")
    missing = [alias for src, alias in wanted if src not in existing]
    if not target_parts:
        missing.append("Typical_Move_Price/Target")
    if missing:
        LOGGER.warning("%s lacks %s -- selected as NULL for signals issued before "
                       "those columns existed", table, ", ".join(missing))

    sql = f"""
    WITH ranked AS (
      SELECT s.*, ROW_NUMBER() OVER (
               PARTITION BY s.Ticker, s.Bar_Date
               ORDER BY s.Run_Timestamp NULLS LAST
             ) rn
      FROM `{project}.{dataset}.{table}` s
      {already}
      WHERE TRUE {where_done}
    )
    SELECT {select}
    FROM ranked WHERE rn = 1
    ORDER BY Bar_Date
    {f'LIMIT {limit}' if limit else ''}
    """
    return client.query(sql).to_dataframe()


def load_evaluated_keys(client: bigquery.Client, project: str,
                        dataset: str) -> set[tuple[str, dt.date, int]]:
    """Every (Ticker, Bar_Date, Horizon_Days) already stored.

    Needed because a signal stays PENDING until its longest horizon completes,
    so the same signal is revisited on every run while its 30-session window
    fills. Without this, each run would re-append the 5/10/15-day rows it
    already wrote and the table would silently accumulate duplicates --
    inflating counts and quietly overweighting whichever signals happened to be
    pending longest.
    """
    ref = f"{project}.{dataset}.{OUTCOME_TABLE_ID}"
    try:
        client.get_table(ref)
    except NotFound:
        return set()
    rows = client.query(
        f"SELECT Ticker, Bar_Date, Horizon_Days FROM `{ref}`"
    ).result()
    return {(r["Ticker"], r["Bar_Date"], int(r["Horizon_Days"])) for r in rows}


def evaluate_signal(row: Any, frame: pd.DataFrame,
                    cost_pct: float) -> list[dict[str, Any]]:
    """Replay one signal against what the stock actually did.

    Returns one dict per horizon that has enough completed sessions. A horizon
    with a partial window is skipped rather than reported short -- a 30-session
    figure computed on 11 sessions is not a conservative estimate of the real
    one, it is a different number wearing its label.
    """
    dates = [pd.Timestamp(i).date() for i in frame.index]
    bar = row.Bar_Date if isinstance(row.Bar_Date, dt.date) else pd.Timestamp(row.Bar_Date).date()
    if bar not in dates:
        return []
    i = dates.index(bar)

    close = frame["Close"].to_numpy(dtype=float)
    high = frame["High"].to_numpy(dtype=float)
    low = frame["Low"].to_numpy(dtype=float)
    open_ = frame["Open"].to_numpy(dtype=float)

    entry_close = close[i]
    # The first genuinely transactable price. The signal is computed FROM the
    # close of bar i, so it does not exist until that session is over; the
    # earliest you can act is the next open.
    entry_open = open_[i + 1] if i + 1 < len(frame) else np.nan
    if not np.isfinite(entry_close) or entry_close <= 0:
        return []

    stop = float(row.Stop_Loss) if pd.notna(row.Stop_Loss) else np.nan
    target = float(row.Typical_Move_Price) if pd.notna(row.Typical_Move_Price) else np.nan

    out: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        window = slice(i + 1, i + 1 + horizon)
        w_close, w_high, w_low = close[window], high[window], low[window]
        if len(w_close) < horizon:
            continue

        stop_idx = np.where(w_low <= stop)[0] if np.isfinite(stop) else np.array([])
        tgt_idx = np.where(w_high >= target)[0] if np.isfinite(target) else np.array([])

        out.append({
            "Ticker": row.Ticker,
            "Bar_Date": bar,
            "Horizon_Days": horizon,
            "Evaluated_At": dt.datetime.now(dt.timezone.utc),
            "Action": row.Action,
            "Setup_Type": row.Setup_Type,
            "Score": _f(row.Score),
            "Expected_Move": _f(row.Expected_Move),
            "Avg_Volatility": _f(row.Avg_Volatility),
            "Tail_Drawdown_Pct": _f(row.Tail_Drawdown_Pct),
            "Spike_Ratio": _f(row.Spike_Ratio),
            "Max_Drawdown_1Y_Pct": _f(row.Max_Drawdown_1Y_Pct),
            "Traded_Value_Cr": _f(row.Traded_Value_Cr),
            "Risk_Pct": _f(row.Risk_Pct),
            "Signal_Price": _f(row.Signal_Price),
            "Stop_Loss": _f(row.Stop_Loss),
            "Typical_Move_Price": _f(row.Typical_Move_Price),
            "Entry_Close": round(float(entry_close), 2),
            "Entry_Open": round(float(entry_open), 2) if np.isfinite(entry_open) else None,
            "Sessions_Available": int(len(w_close)),
            "Return_Close_Pct": _pct(w_close[-1], entry_close, cost_pct),
            "Return_Open_Pct": (_pct(w_close[-1], entry_open, cost_pct)
                                if np.isfinite(entry_open) and entry_open > 0 else None),
            "MFE_Pct": _pct(np.nanmax(w_high), entry_close, cost_pct),
            "MAE_Pct": _pct(np.nanmin(w_low), entry_close, cost_pct),
            "Stop_Hit": bool(len(stop_idx)),
            "Sessions_To_Stop": int(stop_idx[0]) + 1 if len(stop_idx) else None,
            "Target_Hit": bool(len(tgt_idx)),
            "Sessions_To_Target": int(tgt_idx[0]) + 1 if len(tgt_idx) else None,
        })
    return out


def _f(value: Any) -> Optional[float]:
    return float(value) if pd.notna(value) else None


def _pct(price: float, entry: float, cost_pct: float) -> Optional[float]:
    if not (np.isfinite(price) and np.isfinite(entry) and entry > 0):
        return None
    return round((price - entry) / entry * 100 - cost_pct, 3)


def backfill(project: str, dataset: str, table: str, config: S.ScannerConfig,
             limit: int = 0, dry_run: bool = False) -> int:
    client = bigquery.Client(project=project)
    signals = load_pending_signals(client, project, dataset, table, limit)
    if signals.empty:
        print("No signals waiting to be evaluated.")
        return 0

    tickers = sorted(signals["Ticker"].unique())
    print(f"{len(signals)} unevaluated signals across {len(tickers)} tickers "
          f"({signals.Bar_Date.min()} .. {signals.Bar_Date.max()})")

    yf = S.load_yfinance()
    frames: dict[str, pd.DataFrame] = {}
    for batch in S.chunks(tickers, config.chunk_size):
        try:
            data = S.download_batch(yf, batch, config)
        except Exception as exc:
            LOGGER.warning("Batch download failed (%s) -- those tickers skipped this run", exc)
            continue
        for ticker in batch:
            frame = S.get_ticker_frame(data, ticker)
            if not frame.empty:
                frames[ticker] = frame

    done = load_evaluated_keys(client, project, dataset)
    rows: list[dict[str, Any]] = []
    not_ready = missing = duplicate = 0
    for row in signals.itertuples():
        frame = frames.get(row.Ticker)
        if frame is None:
            missing += 1
            continue
        produced = evaluate_signal(row, frame, config.round_trip_cost_pct)
        if not produced:
            not_ready += 1
            continue
        fresh = [r for r in produced
                 if (r["Ticker"], r["Bar_Date"], r["Horizon_Days"]) not in done]
        duplicate += len(produced) - len(fresh)
        rows.extend(fresh)

    by_h = pd.Series([r["Horizon_Days"] for r in rows]).value_counts().sort_index() if rows else pd.Series(dtype=int)
    print(f"produced {len(rows)} outcome rows "
          f"({', '.join(f'{h}d:{n}' for h, n in by_h.items()) if len(by_h) else 'none'})")
    if missing:
        print(f"  {missing} signals had no usable price history")
    if not_ready:
        print(f"  {not_ready} signals have no completed horizon yet -- they stay pending")
    if duplicate:
        print(f"  {duplicate} horizons already stored from an earlier run -- skipped")

    if not rows or dry_run:
        if dry_run:
            print("dry run -- nothing written")
        return len(rows)

    out = pd.DataFrame(rows)
    out["Bar_Date"] = pd.to_datetime(out["Bar_Date"]).dt.date
    ref = f"{project}.{dataset}.{OUTCOME_TABLE_ID}"
    try:
        client.get_table(ref)
    except NotFound:
        client.create_table(bigquery.Table(ref, schema=OUTCOME_SCHEMA))
        print(f"created {ref}")

    client.load_table_from_dataframe(
        out, ref,
        job_config=bigquery.LoadJobConfig(
            schema=OUTCOME_SCHEMA,
            write_disposition="WRITE_APPEND",
            schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION],
        ),
    ).result()
    print(f"wrote {len(out)} rows to {ref}")
    return len(out)


def report(project: str, dataset: str, horizon: int = 15) -> None:
    """Print what the evidence says so far, and refuse to overclaim on thin data."""
    client = bigquery.Client(project=project)
    ref = f"{project}.{dataset}.{OUTCOME_TABLE_ID}"
    try:
        client.get_table(ref)
    except NotFound:
        print(f"{ref} does not exist yet -- run --backfill first.")
        return

    df = client.query(f"SELECT * FROM `{ref}` WHERE Horizon_Days = {horizon}").to_dataframe()
    if df.empty:
        print(f"No outcomes stored at the {horizon}-session horizon yet.")
        return

    print(f"\n{'=' * 78}\nFORWARD TEST -- {horizon} sessions, net of costs")
    print(f"{len(df)} signals, {df.Ticker.nunique()} tickers, "
          f"{df.Bar_Date.min()} .. {df.Bar_Date.max()}\n")

    for label, col in (("entry at signal close (not reachable)", "Return_Close_Pct"),
                       ("entry at NEXT OPEN (tradeable)", "Return_Open_Pct")):
        s = df[col].dropna()
        if s.empty:
            continue
        print(f"  {label:<40} mean {s.mean():+6.2f}%  median {s.median():+6.2f}%  "
              f"win {100 * (s > 0).mean():4.1f}%")
    gap = (df["Return_Close_Pct"] - df["Return_Open_Pct"]).dropna()
    if not gap.empty:
        print(f"  {'cost of the unreachable assumption':<40} {gap.mean():+6.2f}pp")
    print(f"\n  best gain offered (MFE)  mean {df.MFE_Pct.mean():+6.2f}%   "
          f"worst dip (MAE) mean {df.MAE_Pct.mean():+6.2f}%")
    # Reported only over rows that HAVE the level, and the denominator is
    # printed. A rate computed over rows where the level was never stored reads
    # as "never reached" when it means "never set" -- which is how a null column
    # once produced a confident 0.0% target-hit rate.
    for label, hit_col, level_col in (("stop", "Stop_Hit", "Stop_Loss"),
                                      ("target", "Target_Hit", "Typical_Move_Price")):
        have = df[df[level_col].notna()]
        if have.empty:
            print(f"  {label} hit: not measurable -- {level_col} is null on all "
                  f"{len(df)} rows")
        else:
            print(f"  {label} hit {100 * have[hit_col].mean():4.1f}%  "
                  f"(of {len(have)} rows that carry a {label})")

    # Stop and target are only comparable across rows if they MEAN the same
    # thing across rows, and in this table they do not. Both levels were pinned
    # at exactly 3% between 2026-08-26 and 2026-09-15 by the retired
    # rr_floor_stop, then freed. Pooling those with ATR-sized stops produces a
    # confident hit rate that measures the config history rather than the model:
    # "stop hit 66%" was really "hit a 3% stop", which any 5%-range stock does
    # by accident. Rows are grouped by level width so the drift is visible
    # instead of averaged away.
    levels = df.dropna(subset=["Stop_Loss", "Entry_Close"]).copy()
    if not levels.empty:
        levels["stop_w"] = (levels.Stop_Loss / levels.Entry_Close - 1) * 100
        pinned = (levels.stop_w.round(1) == levels.stop_w.round(1).mode().iloc[0]).mean()
        print(f"\n  --- stop/target width, to check they mean the same thing ---")
        print(f"    stop width   p10 {levels.stop_w.quantile(.1):+5.2f}%  "
              f"median {levels.stop_w.median():+5.2f}%  p90 {levels.stop_w.quantile(.9):+5.2f}%")
        if pinned > 0.30:
            print(f"    WARNING: {pinned:.0%} of rows share one stop width. Those signals "
                  f"came from a\n    build that pinned the stop, so Stop_Hit/Target_Hit "
                  f"above describe that old rule,\n    NOT the current model. Returns, MFE "
                  f"and MAE are unaffected -- they use prices only.")

    # The question the whole exercise exists to answer.
    ret = "Return_Open_Pct" if df["Return_Open_Pct"].notna().any() else "Return_Close_Pct"
    print(f"\n  --- does the ranking predict? (using {ret}) ---")
    for col in ("Expected_Move", "Score", "Avg_Volatility", "Spike_Ratio", "Traded_Value_Cr"):
        pair = df[[col, ret]].dropna()
        if len(pair) < 30:
            print(f"    {col:<20} n={len(pair)} -- too few to judge")
            continue
        print(f"    {col:<20} corr {pair[col].corr(pair[ret]):+.3f}   n={len(pair)}")

    if len(df) < 200:
        print(f"\n  NOTE: {len(df)} signals is not enough to conclude anything. These "
              f"correlations\n  will move a lot as data accumulates. Treat them as a "
              f"progress bar, not a result.")
    print()


# Both scanners' tables. The grader must cover BOTH or the freeze produces
# nothing for one of them: swings.py and momentum.py write to separate tables
# since the 2026-09-23 split, and S.DEFAULT_BQ_TABLE_ID is now only the swing
# one. Grading just that would leave momentum -- the scanner with the evidence
# behind it -- completely unmeasured.
GRADED_TABLES = ("swings", "momentum")


def main(request: Any = None) -> Optional[tuple[str, int]]:
    """CLI program, or HTTP Cloud Function when handed a request.

    The HTTP path exists so this can run on a schedule. Without it the whole
    point of freezing the model is lost: signal_outcomes only grows when
    something runs this, and nothing was.

    Same guard as swings.main -- never call parse_args in the HTTP path, because
    sys.argv holds the CONTAINER's flags (functions-framework's own
    --target/--source/--port), not anything about the request.
    """
    if request is not None:
        try:
            body = request.get_json(silent=True) or {}
        except Exception:
            body = {}
        project = S._resolve_project_id(body.get("project_id"))
        dataset = body.get("dataset_id", "data_options")
        tables = body.get("tables") or list(GRADED_TABLES)
        config = S.ScannerConfig()
        graded = []
        for t in tables:
            try:
                backfill(project, dataset, t, config, int(body.get("limit", 0)), False)
                graded.append(t)
            except Exception as exc:
                LOGGER.warning("Forward test failed for table %s: %s", t, exc)
        return (f"Forward test graded: {', '.join(graded) or 'nothing'}", 200)

    p = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backfill", action="store_true", help="evaluate signals whose window has completed")
    p.add_argument("--report", action="store_true", help="analyse stored outcomes")
    p.add_argument("--horizon", type=int, default=15, help="horizon to report on (default 15)")
    p.add_argument("--limit", type=int, default=0, help="cap signals evaluated (testing)")
    p.add_argument("--dry-run", action="store_true", help="compute but do not write")
    p.add_argument("--project-id", default=None)
    p.add_argument("--dataset-id", default="data_options")
    p.add_argument("--table-id", default=None,
                   help=f"single table to grade; default grades all of {GRADED_TABLES}")
    args = p.parse_args()

    if not (args.backfill or args.report):
        p.error("nothing to do -- pass --backfill and/or --report")

    project = S._resolve_project_id(args.project_id)
    config = S.ScannerConfig()

    if args.backfill:
        for table in ([args.table_id] if args.table_id else list(GRADED_TABLES)):
            print(f"--- grading {table} ---")
            backfill(project, args.dataset_id, table, config, args.limit, args.dry_run)
    if args.report:
        report(project, args.dataset_id, args.horizon)


if __name__ == "__main__":
    main()
