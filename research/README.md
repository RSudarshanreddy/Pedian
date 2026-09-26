# research/ — the measurements behind STATE.md

Every number in STATE.md's "Tested and REJECTED" section came from one of these.
They are kept so December does not start by rewriting them, and so any claim
can be re-checked rather than believed.

Each pair is `<name>.py` (scans the universe, writes `<name>.csv`) and
`<name>_report.py` (reads that CSV, prints the tables). Run them from this
directory; the CSVs are intermediate and deliberately not committed.

```bash
cd research
python winners.py && python winners_report.py      # ~6 min, downloads 2y for ~2000 tickers
```

| script | question it answers | what it found |
|---|---|---|
| `winners.py` | Where in the ranking do big winners actually sit? What separates ANTELOPUS/AEGISLOG/SHILPAMED from GANECOS? | The rank gradient is NOT monotone — 4-6 is a dead zone at 0.91x the base rate, 11-15 recovers to 1.61x, 26-40 to 1.56x. Upside dominance does not separate; liquidity expansion is *inverse*. **This is the script behind the 3 → 20 change.** |
| `rank.py` | Can the ranking be made direct — driven by something that changes daily instead of a 60-day median? | 13 daily features ranked in both directions, plus 6 blends. None beat `Expected_Move`. Freshness and return trade off monotonically. |
| `fresh.py` | Does `Setup_Age_Days` predict? Do the setup types? | `age == 1` (+0.88%) is worse than `age 2-3` (+1.11%) — newness carries nothing. `breakout` is the worst setup at 7 sessions (−0.18%, 41.5% win). Also produces the depth table: 23% of sessions have zero actionable names in the top 3. |
| `exits.py` | Trailing stops, take-profits, and the owner's own "sell once up X%" rule. | Every trailing stop loses to no stop. The owner's rule wins 66.9% with a +13.99% median and earns half as much as holding. |
| `fixedstop.py` | Fixed stops from entry, including the scanner's own `Stop_Loss`. | Monotone — tighter stop, worse result, no exceptions. `Stop_Loss` is the worst rule measured: +5.48% vs +10.61% for ignoring it. |
| `accel.py` | The 2026-09-27 review's three untested ideas: acceleration shape (5D>10D>20D), EMA20/EMA50 trend structure, and five gate tightenings. | Acceleration is negative. EMA trend structure is real (+9.10% vs +8.20%) but costs 17.8pp of coverage. Three of the five gate tightenings remove **zero rows**. |

## Conventions every one of these follows

Break any of them and the result is not comparable to STATE.md.

- **No lookahead.** Backward windows close at `i - move_horizon_days`, never at
  `i`. A bug of exactly this kind once inflated a correlation from +0.068 to
  +0.249 before it was caught.
- **Entry at the NEXT open**, never at the decision-day close. The scanner reads
  a completed bar; you cannot trade that bar.
- **Net of `round_trip_cost_pct`** on both ends.
- **Split-half validation on BOTH axes** — disjoint ticker halves (`[::2]`) and
  disjoint time halves. A finding that holds on one and not the other is
  discarded. This is what killed most of the forty ideas.
- **Gates copied from `ScannerConfig`, not hardcoded.** A local default that
  drifted from the deployed one made every local run silently stricter than
  production for a while.
- **Score on what the list is FOR.** `winners.py` and `accel.py` report
  `P(r30 > 25%)` and whether a winner is *present* in the list, not just the
  mean. Optimising the mean is what produced the wrong `display_top_n = 3`.

## Known limits

- One 8-month window, ~163 decision dates, ~278 tickers surviving the 2-year
  history requirement. The live universe is ~2000 and the live pool is ~89/day
  against this backtest's ~50 — they are **not** the same population, and they
  have already disagreed once (STATE.md, December Q4).
- `yfinance` is re-downloaded on every run, so two runs days apart are not
  identical. Nothing here is seeded.
- These measure the scanner, not the trader. The tradebook analysis in STATE.md
  came from `tradebook-GP1379-EQ.csv` exported from Kite, not from these.
