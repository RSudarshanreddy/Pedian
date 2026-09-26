# Pedian — where things stand

Frozen 2026-09-27. Read this before changing anything.

If you are on a new machine: everything needed is in this repo and in GCP.
Nothing important lives in a local scratchpad. The backtest CSVs behind the
numbers below are gone — the numbers are recorded here instead, because that
is what survives.

## What it does

Two scanners, opposite premises, same universe filter. Their picks overlap 10%.

| | buys | fires | evidence |
|---|---|---|---|
| `scanner/src/momentum.py` | what is already moving hardest | most days, ~89 candidates | **solid** — 163 decision dates, top-3 +13.30% over 30 sessions, disjoint ticker halves 19.06 / 8.36 |
| `scanner/src/swings.py` | a stock that fell ≥12% from its 20-session high while still above its 50-day average | ~every other day, ~7 candidates | **provisional** — +10.06% but only 157 trades ≈ 5–6 independent windows |

Both rank the survivors, neither models an exit. Every row carries an entry, a
stop and a target; deciding when to sell is yours.

**Follow momentum. Audit swings.** The swing rule's winning cell is also its
thinnest — the shallower `dip>=8%` variant measured *worse*, which is the shape
of an overfit even though both splits were clean.

## What is running

All in GCP project `sudarshan-442212`, region `europe-west1`,
schedulers in `asia-south1`, all services at **min-instances 0**.

```
09:07  momentum       previous complete session, actionable pre-open
09:10  swings         same, dip list
10:00  momentum       intraday breakouts, same-day entry
14:00  momentum
17:00  forward-test   grades BOTH tables, every weekday
```

BigQuery `data_options`: `momentum`, `swings`, `signal_outcomes`.

Note `swings` holds momentum-era rows up to 2026-09-22 and swing-era rows after.
`Bounce_Median IS NOT NULL` identifies swing rows. Left deliberately unlabelled.

**Momentum runs 3x/day and each run APPENDS ~89 rows under the same Run_Date.**
Any query grouping by `Run_Date` alone triple-counts. Always dedupe:
`ROW_NUMBER() OVER (PARTITION BY Run_Date, Ticker ORDER BY Run_Timestamp DESC) = 1`.
Within one day the runs can also use different `Bar_Date` (Yahoo lag) — on
2026-09-25 the 09:07 run used 09-23 data and a later run used 09-25.

Cost: inside the Cloud Run free tier. The only thing billing is ~504 MB of
stored container images (~₹5/month). `min-instances = 1` was once 95% of a
₹1,069 bill — if a bill appears, check that first.

Deploying momentum needs `cloudbuild.momentum.yaml`, NOT `--source`:
`gcloud run deploy --source` only ever uses the file literally named
`Dockerfile`, which serves swings.py, and would silently ship the swing scanner
under the momentum name.

## The freeze

No config changes, no new strategies, until `signal_outcomes` has enough graded
rows. Everything below rests on 163 backtest dates from one 8-month window, and
that window has already been caught disagreeing with live (see December, Q4).

Exempt: a genuine bug, a data-integrity failure, or the grader breaking.

## What changed 2026-09-26, and why

`display_top_n: 3 → 20`, plus a new display-only floor
`display_min_move_pct = 18.0`. Shipped as `fdf731e`, revision
`momentum-00005-gbl`. Nothing about what is scanned or stored changed.

**The objective was wrong, not the tuning.** 3 had the best average return,
which is the right target only if the list is bought mechanically. It is not —
it is read by eye and picked from. For a reader who selects, the measure is
whether the list *contains* a winner:

| shown | winner present | r30 of the list | share of all big winners |
|---|---|---|---|
| top 3 | 54.0% | +13.30% | 15% |
| top 10 | 67.5% | +6.24% | 28% |
| **top 20** | **82.2%** | **+6.26%** | **43%** |
| top 30 | 82.8% | +6.22% | 54% |

The evidence that forced this was the owner's own tradebook, not a backtest:
**ANTELOPUS +49.8% at rank 5-15, SIGMAADV at rank 12, BLUESTONE at rank 18** —
all bought, all worked, all invisible under a 3-row cut.

The floor at 18 exists because a plain cap dilutes (+6.26% vs +13.30%).
`>=18 AND top 20` returns +8.20% over 30 sessions with a winner present on
78.5% of sessions. 18 also keeps the names that worked (ANTELOPUS 22.4-22.9,
SHILPAMED 23.9-32.6, AEGISLOG 16.5-19.8) and drops GANECOS (14.7, stuck);
floors at 26 or 28 would have removed all three.

### Correction to what this file used to say

It previously read *"Ranks 1 and 2 are the list. Below the cut is not a
near-miss; it is a different quality of name."* That is true for mechanical
buying and false for how the list is actually used. The rank gradient is not
monotone — P(r30 > 25%), base rate 12.0%:

| rank | P(big) | lift | r30 |
|---|---|---|---|
| 1 | 35.0% | 2.90x | +19.01% |
| 2 | 35.6% | 2.96x | +15.92% |
| 3 | 17.2% | 1.43x | +4.98% |
| **4-6** | **10.9%** | **0.91x** | **+1.33%** |
| 7-10 | 13.9% | 1.15x | +4.29% |
| **11-15** | **19.4%** | **1.61x** | **+7.26%** |
| 16-25 | 13.2% | 1.10x | +5.01% |
| **26-40** | **18.8%** | **1.56x** | **+8.68%** |
| 41+ | 7.3% | 0.61x | +2.07% |

Ranks 1-2 are still far and away the best. But 4-6 is a dead zone *below the
base rate*, and 11-15 and 26-40 recover. ANTELOPUS was findable at rank 5-15
because that part of the list genuinely carries winners.

## Tested and REJECTED — do not re-propose

Roughly forty ideas were measured against this scanner on 2026-09-26/27.
**Not one beat ranking by `Expected_Move`.** Recorded so none of it gets
re-derived, including by the assistant.

**Exits** (163 dates, top-3 picks, entry next open, net of costs):

| exit rule | r30 | note |
|---|---|---|
| **no stop at all** | **+10.61%** | best |
| trailing stop 20% | +9.10% | |
| sell once up 15% | +5.47% | 66.9% win, median +13.99% — best *feel*, half the money |
| trailing stop 12% | +0.33% | |
| trailing stop 8% | −1.19% | |
| fixed stop −20% | +10.10% | |
| fixed stop −15% | +8.95% | |
| fixed stop −8% | +5.84% | |
| **the scanner's own `Stop_Loss`** | **+5.48%** | worst on the board, fires 56% of the time |

Every stop, trailing or fixed, at every distance, loses to no stop. Mechanism:
a 12% trail saves 11.8pp on the big losers and costs 38.3pp on the big winners,
and 37.5% of names that finish only −4% dip through −15% first. The scanner
selects for stocks that lurch, so any stop close enough to protect is close
enough to be hit by noise.

**Rankers** (top 3, r15, baseline `Expected_Move` +5.98%):

acceleration shape 5D>10D>20D (+3.13% and winner present on only 30.7% of
sessions — accelerating names did *worse* than non-accelerating), 10-day return
(+3.49%), 5-day return (+3.42%), distance above EMA20 (+3.67%), relative volume
(+1.47%), volume spike (+1.64%), breakout proximity / low pullback (+4.04%),
position in 20-day range (+4.66%), upside dominance (+1.63%), liquidity
expansion (+0.19% — *inverse*: the low-inflow band produced more big winners
than the high-inflow band; ANTELOPUS's 31x liquidity growth was the consequence
of its move, not a predictor), Score (does not predict, −0.065), Action (BUY
+1.36% vs WATCH +3.71% within the top 3).

Blends with `Expected_Move` all lose too: +dist_ema −0.02pp, +r10 −0.32pp,
+breakout proximity −0.58pp, +r5 −1.23pp, +volume −1.49pp.

**Freshness.** The top 3 was identical on 64% of consecutive sessions and
KABRAEXTRU/SETL/INDSWFTLAB held ranks 1-2-3 for four straight days. Every fix
failed: `age == 1` returns +0.88% against `age 2-3`'s +1.11% (newness does not
predict), a FRESH block of age-1 breakout/reclaim returns +1.03% vs +2.29%,
setup-filtered rankings are all worse, and `typ + dist_ema` cut identical
sessions 64% → 36% for a 0.02pp difference but left the *same core names* on
the days that actually mattered. **93.2% of today's pool was in yesterday's
pool** — the pool is what repeats, not the ranker, and re-ranking a
93%-identical pool cannot produce a fresh list.

**Gate tightening.** `volatile days 12 → 15` and `volatility ratio 20% → 25%`
remove **zero rows** — at a 60-day lookback with avg volatility ≥3.5 everything
surviving already has 15+ volatile days. `median volatility 2.5 → 3.0` removes
15 rows of 8,157. `spike ratio 4.5 → 4.0` costs 8pp of coverage. Only
`min_avg_volatility 3.5 → 4.0` is real: pool +4.28% → +4.86%, P(big) 12.0% →
13.1%, coverage unchanged, both half-splits improve — but only +0.23pp on the
20-name list, too small to justify breaking the freeze. **December candidate.**

**EMA trend structure** (EMA20 > EMA50, both rising, price above EMA20) is the
one rejected idea with a real signal: +9.10% vs +8.20% on the 20-name list,
both half-splits strong. Rejected because it costs **17.8pp of coverage**
(winner present 60.7% vs 78.5%) — the identical trade already refused when
turning down the `Expected_Move >= 28` gate. **December candidate.**

## Why the list cannot currently be made "better, not shorter"

The scanner has **one dimension**. `Score`, `Action`, `Setup_Type`,
`Upside_Dominance`, volume and the five volatility variants are either
different views of the same volatility or measure nothing. That is why:

- stacking every available quality gate takes the live pool only from 89 to 43,
  never to 25 — the 89 already passed the volatility gates, so more volatility
  conditions remove almost nothing;
- `Expected_Move` is the only lever that can cut 89 → 25, and an absolute
  threshold on it is unstable (see December, Q4);
- every "tighter gate" is more of the same single variable under another name.

**A second dimension has to come from outside the 60-day OHLCV series** —
sector strength, delivery percentage, index membership, results dates. Nothing
inside the price series worked.

## December — what to review, in order

Run `python scanner/src/forward_test.py --report --horizon 30`, and dedupe by
`Run_Timestamp` in every query (see "What is running").

**Q1. Does the top 20 beat rank 41+ on LIVE signals?** The whole 3 → 20 change
rests on backtested winner-presence. Live should show top-20 carrying ~43% of
big winners at ~2x the base rate, and rank 41+ *below* base rate (0.61x).
If rank 41+ is not worse, the ranking is not working at all.

**Q2. Is the rank 4-6 dead zone real?** Backtest says 0.91x, below base rate,
across two independent horizons and two samples. It is the strangest result in
this file. If it survives live, it is worth understanding.

**Q3. Do 11-15 and 26-40 really recover?** This is what justified showing 20
instead of 3. If the gradient is monotone live, the 3-row cut was right after
all and this change should be reverted.

**Q4. Does `min_typical_move_pct = 26` give a stable ~25 names?** Backtest (278
tickers) says it gives **zero** names on 29% of days. Live (2000 tickers, five
days) gave 27, 27, 26, 25, 25. These flatly disagree and five days cannot
settle it. Count the live pool at gates 22 / 26 / 28 over December. If it is
stable, the owner's "89 is not a scanner" objection can finally be actioned.

**Q5. The two December candidates** — `min_avg_volatility 4.0` and the EMA
trend structure. Both measured better on quality and worse or neutral on
coverage. Decide with live data, not another backtest.

**Q6. Does the steady-compounder blind spot cost anything?** `Expected_Move` is
a median magnitude, so it ranks grinders low on purpose. MANINDS on 2026-09-24
ranked 70 of 86 on 17.4% and did +18.4% in a month, +130.6% in six. If names
like this keep landing at rank 70 while outperforming, the ranker needs a
consistency term. Do NOT add one on backtest evidence.

## What the tradebook says (measured 2026-09-26, Aug 3 – Sep 25)

Real fills, 228 of them, 111 actual decisions over 32 sessions. This corrected
several things the assistant had inferred wrongly from DP-charge ledger lines.

- **48 closed round trips, 89.6% win rate, +₹46,117 net, +6.68% average,
  median hold 10 calendar days.** That is +0.95%/day against +0.35%/day for
  the 30-session hold the assistant had been recommending. The scale-in /
  scale-out on AEGISLOG (12 orders) and ANTELOPUS (11 orders) was the most
  profitable behaviour in the book, not a leak.
- **Median position ₹13,713, largest ₹83,409.** Already correctly sized;
  concentration is not the current problem.
- **The 89.6% is survivorship-flattered** — FIFO only matches trips whose buy
  leg closed, and losers stay open. ₹236,270 (23.6% of the account) was bought
  before 15 Sep and never sold: BSE ₹122,977 held 51 days, E2E ₹86,635 held 47.
  Winners close in 10 days; losers are held indefinitely.
- Gap to ₹50k/month is **utilisation, not strategy**: ~₹23k/month from a
  rotating bucket running at roughly a quarter of capacity.

## Do this regardless

**Stop using the `Stop_Loss` column.** +5.48% against +10.61% for ignoring it,
fires on 56% of trades, sits 7.8% below entry. It is printed on every row and
stored in BigQuery, which makes it look endorsed. It is the worst rule measured.

**Size every position the same.** Equal-weighting the existing positions turned
−₹10,412 into +₹6,506 — larger than any edge either scanner has demonstrated,
and it needs no validation.

## Things known to be wrong

- `Today_Range` reports magnitude, not direction — CUPID showed 7.5% on a day
  it fell 3.32%. Display only, feeds no gate.
- On the swing list `Action` is always BUY and `Setup_Type` always
  `deep_pullback` — both determined by the gate that admitted the row.
- `position_check.py` still replays the retired "sell at +3%" rule under its
  own `LEGACY_*` constants. Its scheduler is PAUSED.
- Yahoo runs 1–3 sessions behind for most of the market. `Bar_Date` records
  which session was actually used — read it rather than assuming it is today.
- `select_for_display`'s fallback (show the pool if the floor empties it) has
  **never fired**: 0 of 163 sessions had no candidate clearing 18%. Harmless,
  but it is guarding an impossible case.
- Circuit-locked names still rank normally. KABRAEXTRU held rank 1 for four
  sessions while locked in upper circuit and un-buyable. Detectable from the
  bar (0.00% range + volume collapse). Deliberately not fixed — the owner's
  view is that removing a UC name is not a fix, and the measurements agree that
  `extended` is the best-performing setup, so it should not simply be dropped.

## Why the config comments are long

Each threshold carries the sweep that set it. They exist so a number is not
re-litigated from intuition. Three findings were retracted in one session for
looking good on one split and failing another; a whole day's worth were
retracted on 2026-09-26. The comments are what stops that repeating.
