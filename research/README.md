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
| `sector_test.py` (session scratchpad, not in the repo) | 2026-10-29: a market gauge per sector, Finance and Energy first. Q1: with the market ON, do picks in a sector at >= 50% beat those below? Q2: with the market OFF, do picks in a sector at >= 50% make money? Picks = dayk's MOVING/SWINGS/FORM listed days, Dec-Sep, each list's plan. | **Neither passes.** Finance was never ON on a market-OFF day (it falls with the market), and its picks were the weakest: −1.56% a trade, 47% won, with market and sector ON. Energy was ON on 34 of 99 market-OFF days, but the lists had 1 energy pick on those days. With the market ON, energy picks did *better* with the sector OFF (+4.40 vs +2.08), against the idea; one time half too thin to call. |
| `sector_all.py` (session scratchpad, not in the repo) | 2026-10-29: the same two questions for every NSE sector plus defence (Nifty India Defence list) and gold (jewellers/bullion). Verdict on the POOLED test (each pick against its own sector's gauge), since ~20 sectors tried means a lucky pass or two. | **Both FAIL.** Q1: sector ON +2.02 vs OFF +2.11 (n 3,107 / 318), halves mixed. Q2, buying a strong sector in a falling market, was the WORST group: −2.93% a trade, 37% won, negative on all four halves (182 picks, 76 stocks); market-OFF picks in weak sectors did +0.78. No single sector passed (most too few). Gold (~5 stocks in range a day) and defence (~9) are too small for a breadth gauge. |
| `sector_wide.py` (session scratchpad, not in the repo) | 2026-10-29, the owner's idea: universe = close > Rs 300 with NO upper cap (547 stocks a day vs 415). Picks rebuilt for all three lists; (B) what the change does per list, gauge ON; (A) the sector gauge again. | **It depends on the list.** SWINGS above 1600 HELPS: +5.00% a trade, 71% won, all four halves positive (161 picks, 30 stocks; +4.66 without the top 3). MOVING above 1600 does NOT: +0.72 vs +2.53 kept. FORM below 300 (its floor is 100) HURTS: −0.19 vs +1.37 kept, worse on all four halves. Dropping 250-300 from MOVING/SWINGS does not help. Market gauge barely moves (same ON/OFF on 96% of days). Sector gauge: Q2 FAILS harder (−5.03%, 31% won); Q1 passes here (+2.27 vs −0.50) but failed on the 250-1600 universe, so not robust. |

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
