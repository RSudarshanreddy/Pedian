"""
POSITION MOMENTUM CHECK
=======================
Separate from swings.py on purpose -- this checks REAL holdings (your actual
entry price/date from data_options.holdings) against the same floor+extension
exit rule the scanner is built on, not scanner candidates. Different concern,
different lifecycle: swings.py finds new trades; this tracks whether trades
you're already in are still worth holding.

Answers the question asked by hand, over and over, for BALUFORGE/HEG/DYCL
this session: is this position still in its momentum window (worth keeping
capital in), or did it already hit the rule's real exit trigger?

Usage:
    python position_check.py                    # check all active holdings
    python position_check.py --project-id XYZ    # override default project
"""

from __future__ import annotations

import argparse
import datetime as dt
from typing import Optional

import pandas as pd
from google.cloud import bigquery

import swings

# Zerodha Console's holdings CSV export uses these headers (case varies by
# export version) -- matched case-insensitively, with a couple of common
# aliases, rather than hardcoding one exact spelling.
CSV_TICKER_COLUMNS = ["instrument", "symbol", "tradingsymbol"]
CSV_QTY_COLUMNS = ["qty.", "qty", "quantity", "quantity available"]
CSV_AVG_COST_COLUMNS = ["avg. cost", "avg cost", "average price", "avg_cost"]


def _find_column(columns: list[str], candidates: list[str]) -> Optional[str]:
    lower_map = {c.lower().strip(): c for c in columns}
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def import_holdings_csv(csv_path: str, project_id: str) -> None:
    """
    Loads a Zerodha Console holdings CSV export into data_options.holdings.

    Zerodha's export has quantity and average cost, but NOT the original
    entry date -- that's a real gap, handled deliberately, not glossed
    over: for a ticker already in holdings, only Quantity/Entry_Price are
    updated, the existing Entry_Date is kept as-is (it's the one piece of
    truth the CSV can't provide). For a ticker not already in holdings,
    Entry_Date is set to today and printed with an explicit warning --
    correct it manually if it wasn't actually bought today, since the
    momentum trace is meaningless from the wrong starting point.

    Any ticker currently Active=TRUE in holdings but NOT present in this
    CSV is treated as closed (the export is a complete current snapshot)
    and marked Active=FALSE.
    """
    df = pd.read_csv(csv_path)
    columns = list(df.columns)

    ticker_col = _find_column(columns, CSV_TICKER_COLUMNS)
    qty_col = _find_column(columns, CSV_QTY_COLUMNS)
    avg_col = _find_column(columns, CSV_AVG_COST_COLUMNS)

    if not (ticker_col and qty_col and avg_col):
        raise ValueError(
            f"Could not find expected columns in {csv_path}. Found: {columns}. "
            f"Need something matching ticker={CSV_TICKER_COLUMNS}, qty={CSV_QTY_COLUMNS}, "
            f"avg_cost={CSV_AVG_COST_COLUMNS}."
        )

    client = bigquery.Client(project=project_id)
    # Full existing rows, not just Ticker/Entry_Date -- closed positions need
    # their Entry_Price/Quantity/Notes carried over too, not just flipped to
    # inactive with data loss.
    existing = {
        row["Ticker"]: dict(row.items())
        for row in client.query(
            f"SELECT Ticker, Entry_Price, Entry_Date, Quantity, Notes "
            f"FROM `{project_id}.data_options.holdings` WHERE Active = TRUE"
        ).result()
    }

    today = dt.date.today().isoformat()
    csv_tickers = set()
    new_rows = []
    for _, row in df.iterrows():
        raw_symbol = str(row[ticker_col]).strip()
        if not raw_symbol or raw_symbol.lower() == "nan":
            continue
        # Zerodha appends "-T" to the symbol for T1 (not-yet-settled) lots --
        # that's a settlement-status marker, not part of the real ticker
        # (confirmed: BLISSGVS-T/E2E-T fail on Yahoo, BLISSGVS/E2E don't).
        if raw_symbol.upper().endswith("-T"):
            raw_symbol = raw_symbol[:-2]
        ticker = swings.to_yahoo_nse_ticker(raw_symbol)
        qty = float(str(row[qty_col]).replace(",", ""))
        avg_cost = float(str(row[avg_col]).replace(",", ""))
        if qty <= 0:
            continue

        csv_tickers.add(ticker)
        if ticker in existing:
            entry_date = str(existing[ticker]["Entry_Date"])
        else:
            entry_date = today
            print(f"NEW position detected: {ticker} -- Entry_Date set to {today}. "
                  f"Correct manually if it wasn't actually bought today.")

        new_rows.append({
            "Ticker": ticker, "Entry_Price": avg_cost, "Entry_Date": entry_date,
            "Quantity": qty, "Active": True, "Notes": None,
        })

    closed_tickers = set(existing) - csv_tickers
    closed_rows = [
        {
            "Ticker": t, "Entry_Price": existing[t]["Entry_Price"], "Entry_Date": str(existing[t]["Entry_Date"]),
            "Quantity": existing[t]["Quantity"], "Active": False, "Notes": existing[t]["Notes"],
        }
        for t in closed_tickers
    ]

    all_touched = csv_tickers | closed_tickers
    if all_touched:
        # DELETE-then-LOAD, not a streaming insert -- streamed rows (and the
        # whole table, briefly) can't be UPDATE'd/DELETE'd for a while after
        # insert_rows_json, which silently broke the "mark closed" step the
        # first time this ran. A LOAD job doesn't have that limitation, same
        # pattern already proven safe in swings.py's write_to_bigquery.
        client.query(
            f"DELETE FROM `{project_id}.data_options.holdings` WHERE Ticker IN UNNEST(@tickers)",
            job_config=bigquery.QueryJobConfig(query_parameters=[
                bigquery.ArrayQueryParameter("tickers", "STRING", list(all_touched))
            ]),
        ).result()

        all_rows = pd.DataFrame(new_rows + closed_rows)
        client.load_table_from_dataframe(
            all_rows, f"{project_id}.data_options.holdings",
            job_config=bigquery.LoadJobConfig(write_disposition="WRITE_APPEND"),
        ).result()

    if closed_tickers:
        print(f"Marked as closed (no longer in CSV): {', '.join(sorted(closed_tickers))}")
    print(f"Imported {len(new_rows)} active holdings from {csv_path}")


def _fifo_open_lots(trades: pd.DataFrame) -> tuple[float, float, str, str, int]:
    """
    FIFO-consumes buy/sell rows (already sorted by execution time) for one
    symbol down to whatever's still open. Returns (qty, weighted_avg_price,
    oldest_lot_date, newest_lot_date, num_open_lots). A sell that exceeds
    the lots this dataframe knows about (a pre-existing position from
    before the tradebook's window) just runs the open-lot list to empty --
    it does NOT go negative, so a heavier sell than tracked buys silently
    zeroes out rather than producing a nonsense answer; the caller is
    expected to sanity-check the resulting qty against the real holding.
    """
    lots: list[list] = []  # [qty, price, date]
    for _, row in trades.iterrows():
        qty = float(row["quantity"])
        if row["trade_type"] == "buy":
            lots.append([qty, float(row["price"]), row["trade_date"]])
        else:
            remaining = qty
            while remaining > 1e-9 and lots:
                consume = min(lots[0][0], remaining)
                lots[0][0] -= consume
                remaining -= consume
                if lots[0][0] <= 1e-9:
                    lots.pop(0)
    total_qty = sum(l[0] for l in lots)
    if total_qty <= 1e-9:
        return 0.0, 0.0, "", "", 0
    wavg_price = sum(l[0] * l[1] for l in lots) / total_qty
    return total_qty, wavg_price, min(l[2] for l in lots), max(l[2] for l in lots), len(lots)


def apply_tradebook_entry_dates(tradebook_csv: str, project_id: str) -> None:
    """
    Zerodha Console's holdings CSV (used by import_holdings_csv) has no
    purchase date, so anything new to tracking gets Entry_Date defaulted
    to today -- a real gap, since the momentum/stop/time-stop checks in
    check_position_momentum can't trace history from a fabricated date.

    A Zerodha tradebook export (every individual fill, with real dates)
    can recover the real date for a currently-open position: FIFO-consume
    each symbol's buys/sells down to what's still open, and use the
    OLDEST remaining lot's date as the entry date -- the date the
    currently-held tranche started being accumulated.

    Only touches holdings rows whose Entry_Date is still today (the
    "unknown" marker this same script wrote); rows with a real date
    already (manually seeded, or fixed in an earlier run) are left alone.
    Some positions won't fully reconcile against this file (pre-existing
    shares from before the tradebook's window, e.g. Zerodha's own "long
    term" quantity flag) -- when the FIFO qty is less than the actual
    holding, the true entry is even older than what's found here, so the
    oldest resolvable lot is still a safe (if approximate) choice: it
    understates how long the capital has been deployed, never overstates
    it. Symbols with NO trades in this file at all (bought entirely
    outside its window) are left untouched and reported as unresolved.
    """
    trades = pd.read_csv(tradebook_csv)
    trades["order_execution_time"] = pd.to_datetime(trades["order_execution_time"])
    trades = trades.sort_values("order_execution_time")

    client = bigquery.Client(project=project_id)
    today = dt.date.today().isoformat()
    holdings = {
        row["Ticker"]: dict(row.items())
        for row in client.query(
            f"SELECT Ticker, Entry_Price, Entry_Date, Quantity, Notes "
            f"FROM `{project_id}.data_options.holdings` WHERE Active = TRUE"
        ).result()
    }

    updates = []
    for symbol, g in trades.groupby("symbol"):
        ticker = swings.to_yahoo_nse_ticker(symbol)
        if ticker not in holdings or str(holdings[ticker]["Entry_Date"]) != today:
            continue
        fifo_qty, _, oldest_date, _, _ = _fifo_open_lots(g)
        if fifo_qty <= 1e-9:
            continue
        held_qty = float(holdings[ticker]["Quantity"])
        note = "" if abs(fifo_qty - held_qty) < 0.5 else f" (approx -- tradebook shows {fifo_qty:.0f} vs {held_qty:.0f} held, some shares predate this file)"
        updates.append((ticker, oldest_date, note))

    if updates:
        client.query(
            f"DELETE FROM `{project_id}.data_options.holdings` WHERE Ticker IN UNNEST(@tickers)",
            job_config=bigquery.QueryJobConfig(query_parameters=[
                bigquery.ArrayQueryParameter("tickers", "STRING", [t for t, _, _ in updates])
            ]),
        ).result()
        new_rows = pd.DataFrame([
            {
                "Ticker": t, "Entry_Price": holdings[t]["Entry_Price"], "Entry_Date": new_date,
                "Quantity": holdings[t]["Quantity"], "Active": True, "Notes": holdings[t]["Notes"],
            }
            for t, new_date, _ in updates
        ])
        client.load_table_from_dataframe(
            new_rows, f"{project_id}.data_options.holdings",
            job_config=bigquery.LoadJobConfig(write_disposition="WRITE_APPEND"),
        ).result()

    for t, new_date, note in updates:
        print(f"Corrected {t}: Entry_Date -> {new_date}{note}")

    corrected = {t for t, _, _ in updates}
    still_unresolved = sorted(
        t for t, h in holdings.items()
        if str(h["Entry_Date"]) == today and t not in corrected
    )
    if still_unresolved:
        print(f"\nStill unresolved (no usable trades in this file -- position predates it "
              f"or was entirely offset within it): {', '.join(still_unresolved)}")
    print(f"\nUpdated {len(updates)} Entry_Date(s) from {tradebook_csv}")


def check_position_momentum(
    ticker: str, entry_price: float, entry_date: str, config: swings.ScannerConfig
) -> dict:
    """
    Traces the actual day-by-day price action from a REAL entry point (not
    a scanner-signal day) using the exact same rule run_barrier_backtest
    simulates (see _simulate_floor_extension_trade) -- not just the
    floor+extension part, the full two-phase rule with both backstops:

    Phase 1 (before the floor is reached): walks forward up to
    config.max_hold_days sessions looking for the stop or the floor,
    whichever comes first. A position that never reaches the floor within
    that window is a TIME STOP -- capital that isn't working doesn't get
    to sit indefinitely just because it hasn't technically lost yet.

    Phase 2 (after the floor is reached): same floor+extension logic as
    before -- hold up to config.max_extension_days more sessions while
    still closing higher, exit on the first non-higher close -- but the
    stop stays active as a hard backstop throughout, same as phase 1.

    The stop itself is the same ATR/structural/RR-floor stop
    add_indicators computes for a fresh scanner signal, evaluated as of
    the entry day (no lookahead). This was the real gap before: a holding
    that never reached the floor had no exit trigger at all, no matter how
    far it fell (confirmed live -- IRCTC/AGOL/WAAREERTL were sitting at
    -40% to -60% and still just read "NOT YET AT FLOOR").

    Returns a dict; does not raise on missing data (returns a NO DATA
    verdict instead), since this is meant to run across many holdings
    unattended.
    """
    yf = swings.load_yfinance()
    try:
        raw = yf.download(ticker, period="6mo", interval="1d", progress=False, auto_adjust=True, threads=False)
        data = swings.normalize_single_ticker_columns(raw)
    except Exception as exc:
        return {"Ticker": ticker, "Verdict": f"DATA ERROR: {exc}"}

    # Some symbols (e.g. AGOL) aren't available on NSE via Yahoo but resolve
    # fine on BSE -- confirmed via direct testing (AGOL.NS empty, AGOL.BO ok).
    if data.empty and ticker.upper().endswith(".NS"):
        bse_ticker = ticker[:-3] + ".BO"
        try:
            raw = yf.download(bse_ticker, period="6mo", interval="1d", progress=False, auto_adjust=True, threads=False)
            data = swings.normalize_single_ticker_columns(raw)
            if not data.empty:
                ticker = bse_ticker
        except Exception:
            pass

    if data.empty:
        return {"Ticker": ticker, "Verdict": "NO DATA since entry date"}

    indexed = swings.add_indicators(data, config)
    entry_ts = pd.Timestamp(entry_date)
    post_entry = indexed.loc[indexed.index >= entry_ts]
    if post_entry.empty:
        return {"Ticker": ticker, "Verdict": "NO DATA since entry date"}

    closes = post_entry["Close"].to_numpy()
    lows = post_entry["Low"].to_numpy()
    dates = post_entry.index
    n = len(closes)
    current_price = float(closes[-1])
    pnl_pct = (current_price - entry_price) / entry_price * 100
    floor = entry_price * (1 + config.target_pct / 100)

    stop_raw = post_entry["trade_stop"].iloc[0]
    stop = float(stop_raw) if pd.notna(stop_raw) else None

    base = {
        "Ticker": ticker, "Entry_Price": entry_price, "Entry_Date": entry_date,
        "Current_Price": round(current_price, 2), "PnL_Pct": round(pnl_pct, 2),
        "Days_Since_Entry": n, "Stop": round(stop, 2) if stop is not None else None,
    }

    def _stop_hit_verdict(idx: int, during_extension: bool) -> dict:
        exit_price = float(lows[idx])
        exit_pnl = (exit_price - entry_price) / entry_price * 100
        suffix = " during extension" if during_extension else ""
        return {
            **base,
            "Verdict": (
                f"STOP HIT on {dates[idx].date()} at {exit_price:.2f} ({exit_pnl:+.2f}%) "
                f"-- exit, downside protection triggered{suffix}"
            ),
            "Still_In_Momentum": False,
        }

    # Phase 1: entry day itself (index 0) isn't monitored -- the position
    # starts being watched the day after, same convention as the backtest.
    # Unlike the backtest's bounded trial window, this search is NOT capped
    # at max_hold_days -- a live position that converts a little late still
    # converted, and should move into normal phase-2 tracking, not get
    # frozen into a timeout verdict it already grew out of (a position that
    # crossed the floor on day 6 instead of day 5 is not still "stuck").
    # max_hold_days is only used below to judge "stuck as of today."
    hold = config.max_hold_days
    floor_day_idx = None
    for j in range(1, n):
        if stop is not None and lows[j] <= stop:
            return _stop_hit_verdict(j, during_extension=False)
        if closes[j] >= floor:
            floor_day_idx = j
            break

    if floor_day_idx is None:
        if n - 1 >= hold:
            base["Verdict"] = (
                f"TIME STOP -- {hold}+ sessions since entry, never reached the "
                f"{config.target_pct}% floor -- exit, capital isn't converting"
            )
            base["Still_In_Momentum"] = False
        elif pnl_pct <= -config.max_risk_pct:
            # Catches positions whose Entry_Date is unreliable (CSV import
            # defaults it to today for anything not already tracked, since
            # Zerodha's export has no real purchase date -- see
            # import_holdings_csv) so day-count logic above can't see their
            # real history. PnL_Pct itself is still accurate (it comes
            # straight from Average Price), so a loss past the scanner's
            # own max-acceptable-risk threshold is flagged regardless.
            base["Verdict"] = (
                f"DEEP LOSS -- {pnl_pct:+.2f}% exceeds max acceptable risk "
                f"({config.max_risk_pct}%) -- review for exit"
            )
            base["Still_In_Momentum"] = False
        else:
            base["Verdict"] = "NOT YET AT FLOOR"
        return base

    # Phase 2: extension window, stop still active throughout.
    base["Floor_Crossed_On"] = str(dates[floor_day_idx].date())
    ext_end = min(floor_day_idx + 1 + config.max_extension_days, n)
    prev_close = closes[floor_day_idx]
    for k in range(floor_day_idx + 1, ext_end):
        if stop is not None and lows[k] <= stop:
            return _stop_hit_verdict(k, during_extension=True)
        if closes[k] <= prev_close:
            exit_price = float(closes[k])
            exit_pnl = (exit_price - entry_price) / entry_price * 100
            base["Verdict"] = (
                f"MOMENTUM BROKEN on {dates[k].date()} at {exit_price:.2f} "
                f"({exit_pnl:+.2f}%) -- rule's exit already triggered, even though "
                f"current price may differ"
            )
            base["Still_In_Momentum"] = False
            return base
        prev_close = closes[k]

    if ext_end - floor_day_idx - 1 >= config.max_extension_days:
        base["Verdict"] = f"EXTENSION WINDOW EXPIRED ({config.max_extension_days} days past floor) -- rule says exit now"
        base["Still_In_Momentum"] = False
    else:
        base["Verdict"] = "STILL IN MOMENTUM -- hold"
        base["Still_In_Momentum"] = True
    return base


def book_profit(
    ticker: str, exit_qty: float, exit_price: float, project_id: str,
    exit_date: Optional[str] = None, notes: Optional[str] = None,
) -> dict:
    """
    Records a real sell against an active holding: logs the realized trade
    to data_options.realized_trades (append-only ledger -- the "position
    ledger" gap), then updates data_options.holdings -- full exit closes
    the row (Active=FALSE), partial exit reduces Quantity and keeps the
    original Entry_Price/Entry_Date for the remaining shares (cost basis
    isn't reaveraged on a partial sell).
    """
    ticker = swings.to_yahoo_nse_ticker(ticker.strip())
    client = bigquery.Client(project=project_id)

    rows = list(client.query(
        f"SELECT Ticker, Entry_Price, Entry_Date, Quantity, Notes "
        f"FROM `{project_id}.data_options.holdings` WHERE Ticker = @ticker AND Active = TRUE",
        job_config=bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("ticker", "STRING", ticker)
        ]),
    ).result())
    if not rows:
        raise ValueError(f"No active holding found for {ticker} in data_options.holdings")

    current = dict(rows[0].items())
    entry_price = float(current["Entry_Price"])
    entry_date = str(current["Entry_Date"])
    held_qty = float(current["Quantity"])
    if exit_qty > held_qty:
        raise ValueError(f"Exit qty {exit_qty} exceeds held qty {held_qty} for {ticker}")

    exit_date = exit_date or dt.date.today().isoformat()
    pnl_amount = (exit_price - entry_price) * exit_qty
    pnl_pct = (exit_price - entry_price) / entry_price * 100

    trade_row = pd.DataFrame([{
        "Ticker": ticker, "Entry_Price": entry_price, "Entry_Date": entry_date,
        "Exit_Price": exit_price, "Exit_Date": exit_date, "Quantity": exit_qty,
        "PnL_Amount": round(pnl_amount, 2), "PnL_Pct": round(pnl_pct, 2), "Notes": notes,
    }])
    client.load_table_from_dataframe(
        trade_row, f"{project_id}.data_options.realized_trades",
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_APPEND"),
    ).result()

    remaining_qty = held_qty - exit_qty
    client.query(
        f"DELETE FROM `{project_id}.data_options.holdings` WHERE Ticker = @ticker",
        job_config=bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("ticker", "STRING", ticker)
        ]),
    ).result()
    if remaining_qty > 0:
        updated_row = pd.DataFrame([{
            "Ticker": ticker, "Entry_Price": entry_price, "Entry_Date": entry_date,
            "Quantity": remaining_qty, "Active": True, "Notes": current["Notes"],
        }])
        client.load_table_from_dataframe(
            updated_row, f"{project_id}.data_options.holdings",
            job_config=bigquery.LoadJobConfig(write_disposition="WRITE_APPEND"),
        ).result()

    result = {
        "Ticker": ticker, "Exit_Qty": exit_qty, "Remaining_Qty": remaining_qty,
        "Entry_Price": entry_price, "Exit_Price": exit_price,
        "PnL_Amount": round(pnl_amount, 2), "PnL_Pct": round(pnl_pct, 2),
    }
    print(
        f"Booked {exit_qty} of {ticker} @ {exit_price:.2f} (entry {entry_price:.2f}): "
        f"{pnl_amount:+.2f} ({pnl_pct:+.2f}%). "
        + (f"{remaining_qty} left, still active." if remaining_qty > 0 else "Position closed.")
    )
    return result


def run_position_check(project_id: str, config: Optional[swings.ScannerConfig] = None) -> pd.DataFrame:
    """
    Reads active rows from data_options.holdings, runs check_position_momentum
    on each, prints a summary, and writes the results to
    data_options.position_momentum_check (timestamped, append-only history --
    so "was I told to exit this three days ago" is answerable later, not
    just "what does it say right now").
    """
    config = config or swings.ScannerConfig()
    client = bigquery.Client(project=project_id)

    holdings = list(client.query(
        f"SELECT Ticker, Entry_Price, Entry_Date, Quantity, Notes "
        f"FROM `{project_id}.data_options.holdings` WHERE Active = TRUE"
    ).result())

    if not holdings:
        print("No active holdings in data_options.holdings.")
        return pd.DataFrame()

    checked_at = dt.datetime.now(dt.timezone.utc).isoformat()
    rows = []
    print(f"\n{'Ticker':<14} {'Entry':>9} {'Current':>9} {'PnL%':>7} {'Verdict'}")
    print("-" * 100)
    for h in holdings:
        result = check_position_momentum(h["Ticker"], float(h["Entry_Price"]), str(h["Entry_Date"]), config)
        result["Checked_At"] = checked_at
        result["Quantity"] = h["Quantity"]
        rows.append(result)
        pnl = result.get("PnL_Pct")
        current = result.get("Current_Price")
        print(f"{result['Ticker']:<14} {h['Entry_Price']:>9.2f} "
              f"{current if current is not None else '-':>9} "
              f"{(f'{pnl:+.2f}' if pnl is not None else '-'):>7} {result['Verdict']}")

    results_df = pd.DataFrame(rows)
    try:
        # ALLOW_FIELD_ADDITION -- this run added Stop, a new column the
        # existing table's schema doesn't have yet.
        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND",
            schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION],
        )
        client.load_table_from_dataframe(
            results_df, f"{project_id}.data_options.position_momentum_check", job_config=job_config
        ).result()
        print(f"\nWrote {len(results_df)} rows to {project_id}.data_options.position_momentum_check")
    except Exception as exc:
        print(f"\nposition_momentum_check write skipped/failed: {exc}")

    return results_df


def format_position_telegram_summary(results_df: pd.DataFrame) -> Optional[str]:
    """
    Plain text, same reasoning as swings.format_telegram_digest -- setup/
    verdict text here is data-driven, not worth risking a Markdown parse
    error over. Only worth sending if something needs a decision: skips
    positions that are just "NOT YET AT FLOOR" (nothing to act on) and
    sends nothing at all if every position is in that state.
    """
    if results_df.empty:
        return None

    actionable = results_df[results_df["Verdict"] != "NOT YET AT FLOOR"]
    if actionable.empty:
        return None

    lines = ["Position check -- action needed or momentum update:", ""]
    for _, r in actionable.iterrows():
        lines.append(f"{r['Ticker']}: {r['PnL_Pct']:+.2f}% -- {r['Verdict']}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Check real holdings against the floor+extension exit rule.")
    parser.add_argument("--project-id", default="sudarshan-442212")
    parser.add_argument("--import-csv", default=None,
                         help="Path to a Zerodha Console holdings CSV export -- imports into "
                              "data_options.holdings before running the check")
    parser.add_argument("--tradebook-csv", default=None,
                         help="Path to a Zerodha tradebook export -- recovers real Entry_Date "
                              "(via FIFO) for holdings rows still defaulted to today")
    parser.add_argument("--no-telegram", action="store_true", help="Skip sending the Telegram summary")
    parser.add_argument("--book-profit-ticker", default=None, help="Ticker to book a sell against, e.g. DYCL")
    parser.add_argument("--book-profit-qty", type=float, default=None, help="Quantity sold")
    parser.add_argument("--book-profit-price", type=float, default=None, help="Actual sell price")
    parser.add_argument("--book-profit-notes", default=None, help="Optional note for the realized_trades row")
    args = parser.parse_args()

    if args.import_csv:
        import_holdings_csv(args.import_csv, args.project_id)

    if args.tradebook_csv:
        apply_tradebook_entry_dates(args.tradebook_csv, args.project_id)

    if args.book_profit_ticker or args.book_profit_qty or args.book_profit_price:
        if not (args.book_profit_ticker and args.book_profit_qty and args.book_profit_price):
            raise SystemExit("--book-profit-ticker, --book-profit-qty, and --book-profit-price must all be given together")
        book_profit(
            args.book_profit_ticker, args.book_profit_qty, args.book_profit_price,
            args.project_id, notes=args.book_profit_notes,
        )

    results_df = run_position_check(args.project_id)

    if not args.no_telegram:
        import os
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not bot_token or not chat_id:
            print("\nTelegram summary skipped: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set.")
        else:
            text = format_position_telegram_summary(results_df)
            if text is None:
                print("\nNothing actionable -- Telegram summary skipped.")
            else:
                try:
                    swings.send_telegram_notification(text, bot_token, chat_id)
                    print("\nSent Telegram position summary.")
                except Exception as exc:
                    print(f"\nTelegram summary skipped/failed: {exc}")


if __name__ == "__main__":
    main()
