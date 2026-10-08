# pet

## Now: `form.py`, movers in form, entered on a healthy pullback (2026-10-08)

The owner's model: **find stocks that are in form now, identify their active
momentum leg, wait for a healthy pullback, and catch the next swing.** The
scanner finds the stock, the pullback finds the entry, and the exit manages
the trade.

### The rules

Fixed on 2026-10-08, **before testing**. Every measure uses only the **last 25
sessions**.

**Universe:** NSE stocks priced at ₹100 or more, trading at least ₹10 crore a
day on average over 20 sessions.

**The leg:** from the lowest close in the 25-session window to the highest
close after it. It can be any length. Measured on it:

| measure | meaning |
|---|---|
| size | peak ÷ start − 1 |
| length | sessions from start to peak |
| persistence | share of up-closes during the leg |
| biggest day | the best single day's share of the move |
| gap share | the share of the move that came from gap-up opens |
| inner dip | the deepest drop inside the leg |
| volume | average volume during the leg ÷ the 25-session average |
| acceleration | the second half's return minus the first half's (shown, not filtered) |

**Form**, checked in this order:
1. **Noise:** the leg is under 10%, or it peaked on the window's first day.
2. **Burst / false momentum:** the leg is under 5 sessions, or the biggest day is
   50%+ of the move, or gaps are 50%+ of the move.
3. **Exhausted:** more than 12% below the peak, or more than 10 sessions since
   the peak, or a close at or below the leg's halfway point since the peak.
4. **Emerging:** a leg of 10–15%.
5. **Persistent, the target:** a leg of 15%+, 5+ sessions long, 55%+ up-closes,
   biggest day under 40%, gaps under 40%, inner dip shallower than −10%, and
   volume during the leg at or above average. A 15%+ leg that fails these
   checks counts as burst.

**Candidate (ready to enter):** persistent, **and** the pullback is healthy:
- 4–7% below the peak;
- 1–5 sessions since the peak;
- no day in the pullback worse than −5% (a stand-in for a news shock);
- pullback volume lower than leg volume;
- every close since the peak above the leg's halfway point.

**Ranking:** leg size × persistence. At most 30 names are shown.

**Test trade:** buy at the next open. Exit on whichever comes first:
- the leg's peak price (a limit order, from the day after buying);
- a close below the halfway point (sell at the next open);
- after 10 sessions (sell at the close).

Costs are 0.4% per round trip.

**Judged on:** recent windows, month by month for the last 12 months. It is
compared with the same pullback entry on stocks *without* the in-form filter,
which shows whether finding the mover adds anything.

### First test, 2026-10-08 (`form_test.py`, signals from Oct 2025 to Sep 2026)

| | trades | win % | average | median | avg win | avg loss | hit target | stopped |
|---|---|---|---|---|---|---|---|---|
| **model** (persistent + healthy pullback) | 485 | **58%** | **+0.3%** | **+3.2%** | +4.7% | −5.9% | 54% | 24% |
| control (same pullback, not persistent) | 1,893 | 51% | −0.1% | +0.3% | +4.6% | −4.9% | 46% | 39% |

- **The in-form filter adds value.** With the same entry, persistent movers won
  more often (58% against 51%) and were stopped out far less (24% against 39%).
  The model was ahead of the control in **9 of 12 months**, and ahead of an
  ordinary stock in 8 of 12.
- **It depends on the market.** Every month from Apr to Sep 2026 was positive
  (+0.5% to +2.1% per trade). Oct 2025 to Mar 2026 was mostly negative, with
  Jan 2026 the worst at −3.6%.
- **The edge per trade is small** (+0.3% on average), because losses (−5.9%) run
  larger than wins (+4.7%). The typical trade works (median +3.2%), but the
  losing ones cost more.
- **It gives a short list that changes:** a median of 3 candidates a day (at
  most 15), none on 16% of days, and about 2 new names a day.

The rules were fixed before this test. Changing them now to improve these
numbers would be fitting to the result. Any improvement should be a new,
written-down idea, tested again.

---

## Tried and deleted: `movers.py`, fresh movers among qualified momentum stocks (built 2026-10-01, deleted 2026-10-06)

It was deleted at the owner's request after the replay below showed it did
worse than the frozen list. The findings are kept here so the idea is not
rebuilt.

The owner's best trades (AEGISLOG, ANTELOPUS, SHILPAMED, DIFFNKG) were picked
from the momentum scanner's list in late August, when about 10 new names
entered its top 20 each day. Once the list was ranked by Expected_Move, about 1
new name entered per day, and 17 of the 20 shown names stayed unchanged from
22 Sep onwards. There was nothing fresh to pick from.

`movers.py` keeps the scanner's qualification and changes only the order:

1. **Qualified:** the frozen scanner's own daily pool (~97 names), read from
   BigQuery. In its backtest its gates lifted the 30-session return from
   +0.87% to +4.46%. The scanner itself is untouched.
2. **Fresh:** that pool re-ranked by 5-session gain. About 8 names a day are NEW.
3. **Clean:** falling knives, circuit-locked names and stale prices are removed.

It shows 20 names (never more than 30). It isn't proven yet: the scanner's
backtest found that 5-day return earned less than Expected_Move when buying the
top 3 mechanically. Picking by eye from 20 was never measured, so the monthly
journal (`../journal/report.py`) is the judge.

### Replayed on the scanner's real stored pools, 2026-10-06: the re-ranking is worse

Each run day from 23 Aug to 24 Sep was replayed using exactly the names the
scanner qualified that day (BigQuery `swings` and `momentum`). Each group was
bought at the next open, with 0.4% costs.

| next... | frozen top 20 (Expected_Move) | movers top 20 (5-day re-rank) | whole pool |
|---|---|---|---|
| 5 sessions (27 run days, to the 26 Sep list) | **+1.51%** | +0.23% | +0.60% |
| 10 sessions (20 run days) | **+3.23%** | +2.38% | +2.13% |
| every list from 21 Sep to 2 Oct, held to the 5 Oct close | ahead on **11 of 11** days | | |

The frozen ranking was ahead of the re-ranking on 20 of 27 days over 5
sessions. Every list from 21 Sep to 2 Oct also beat both the re-ranking and the
whole pool when held to the 5 Oct close. That agrees with the scanner's own
8-month backtest. Yahoo had not published 5 Oct yet, so that day's bar came
from Kite. Kite's previous close matched Yahoo's 1 Oct close for all 106 stocks.

- **Fresh names were not better.** Names new to the frozen top 20 made +2.90% over
  10 sessions, against +3.54% for names already there. Names new to the pool made
  +1.90%, against +3.15% for names already there.
- **The pool is where the owner's money came from.** Of 42 buys since 23 Aug,
  the 19 taken from the scanner's pool were up +4.7% on average 10 sessions
  later, and the 23 from outside it +0.1%. Buys from the frozen top 20 were up
  +5.6% (16 buys).
- **The falling-knife flag inside the frozen top 20:** no difference over 5
  sessions (+1.53% against +1.55%). Over 10 sessions knives made +1.46% against
  +3.63%, behind on 12 of 17 days. That's a caution worth watching, not proven.

So the list that felt stagnant was the better list. Its names kept paying, and
new names did worse. These are about 25 overlapping days in one market phase,
but they point the same way as the backtest. **Do not use the 5-day re-ranking.**

---

## Earlier: buy the dip in a range (failed, below)

## The idea

Some stocks swing up and down inside a range for weeks, making several Vs on
the chart. Buy when the price dips to the bottom of the range; sell near the
top. Pure price and volume: no results, no news, no opinions.

## The rules

Fixed on 2026-09-27, **before any testing**. They are checked once a day after
the close, and orders go in for the next day.

**What counts as a V.** A fall of at least 10% (from one day's high to a later
day's low), followed by a rise of at least 10%. A low only counts as a V once
that 10% rise has actually happened, never before.

**A stock qualifies when all of these are true at today's close:**

1. **Active:** its average daily traded value over the last 20 sessions is at least ₹10 crore.
2. **Two Vs:** it has two completed Vs, both within the last 60 sessions (about 3 months). The current dip would be the third.
3. **Floor holds:** the newer V's low is at or above the older one's.
4. **Falling from the ceiling:** its last completed swing was a peak, and the price is coming down from it.
5. **Floor not broken:** today's close is at or above the floor.

The **floor** is the newer V's low. The **ceiling** is the peak after it. The
**range** is the ceiling minus the floor.

**Buy:** a limit order at the floor plus ¼ of the range, for the next day. If
the stock opens below that price, the order fills at the open. Only one trade
is taken per dip.

**Sell** on whichever of these comes first:
- **Target:** the floor plus ¾ of the range, as a limit order from the day after buying.
- **Stop:** a close below the floor minus ¼ of the range. Sell at the next open.
- **Time:** sell at the close of the 15th session after buying.

Every level is measured in quarters of the stock's own range, so a stock that
swings 10% and one that swings 30% are treated the same way. No number is tuned
per stock.

**Costs:** 0.4% per round trip, covering STT, charges and slippage.

## How it is tested

- Every NSE stock (about 2,580) over 4 years of daily prices, replayed one day
  at a time. On each day the rules only see what was known at that day's close.
- It is judged on every trade, not on hand-picked examples.
- It must hold up on both halves of the time period and on both halves of the
  stock list.
- A 10-slot portfolio is compared with holding the Nifty 50 and a Nifty 200
  Momentum 30 fund over the same dates.
- **The rules do not change after the result is seen.** If it fails, it fails.

## Result, 2026-09-27: it failed

Tested once, with the rules exactly as written above: 2,269 stocks, with entries
from 25 Oct 2022 to 22 Sep 2026.

**Per trade** (6,128 closed trades, after costs):

| how the trade ended | share | average |
|---|---|---|
| reached the target | 40% | +9.35% |
| stopped out | 31% | −8.51% |
| ran out of time | 29% | −1.52% |
| **all trades** | | **+0.65%** (median +0.66%, 51% winners) |

The small edge is mostly in the first half: +1.16% up to Nov 2024, then +0.15%
after it, when the median trade lost money. The two halves of the stock list
agree (+0.55% and +0.76%). How liquid the stock was made no difference: every
liquidity fifth returned between +0.4% and +0.9%.

**As an account** with 10 equal slots, taking trades as they came (1,044 trades,
average +0.03%):

| | CAGR | total | worst drawdown |
|---|---|---|---|
| **pet** | **−2.2%** | **−8.5%** | **−42.7%** |
| Nifty 50 | +7.2% | +31.1% | −15.8% |
| Nifty 200 Momentum 30 ETF | +11.8% | +54.8% | −30.8% |

By calendar year, pet made +12.5% in 2023, +10.1% in 2024, −24.5% in 2025 and
+1.0% in 2026 up to September. (These account figures were corrected
after a bug fix in `portfolio()`; they moved by 0.2 points at most.)

**Why the account made less than the average trade.** Signals bunch up in good
markets: 2,398 of them came in 2024 alone. A 10-slot account can't take a whole
bunch at once, so it takes trades evenly through time, bad stretches included.

**What it means.** When a range holds, buying the dip works: those trades made
+9.35%. But on the day you buy, a dip that turns into a V looks the same as a
dip that keeps falling, and 60% of the dips never reached the target.

### Follow-up: would selling sooner have fixed it? No.

The owner's point was that a swing doesn't last: it moves up, then cools down.
That is partly true. Of the trades that ran out of time, **60% had been up 5% or
more** before they faded back to −1.52%. But of the stopped trades, **56% never
got even +2%**; they just kept falling.

`hindsight.py` replays the same entries with 15 different exits: a +3%, +5%,
+7% or +10% target, or the range target, each held for at most 5, 10 or 15 days.

| exit | win % | avg trade | account CAGR | worst drawdown |
|---|---|---|---|---|
| best: +10%, 15 days | 50% | +0.59% | +6.5% | −34% |
| +3%, 15 days | 74% | +0.39% | −4.3% | −45% |
| any exit held 5 days | 50–64% | +0.3 to +0.5% | −11% to −17% | −50% to −58% |

**Even the best exit, chosen with hindsight, is below just holding the Nifty 50**
(+7.2%, −15.8% drawdown). No exit fixes these entries, because the problem is
the entry: about half the dips keep falling.

Two lessons came out of this:
- **Winning more often is not the same as making money.** Selling at +3% wins
  74% of trades and still loses money as an account.
- **The account result is fragile.** Small changes move it by several percent
  a year (+7%/15d gives +3.3%, +5%/15d gives −8.3%), so the ranking between
  exits is mostly luck.

**Data problems noticed.** A few of the worst trades are probably corporate
actions that Yahoo did not adjust for: QUESS fell 50.6% in one day in Apr 2025,
and PARAS 49.1% in Jan 2025. There are too few of them to change the result.

## Known limits

- The test only uses stocks that are listed today. Stocks that were delisted,
  often the worst ones, are missing, and that makes the results look better
  than they really are.
- The data comes from Yahoo's daily prices, which have occasional bad prints.
- Tax is not included in the trade returns. Short-term gains are taxed at 20%.

## Files

| file | what it does |
|---|---|
| `data.py` | Fetches the NSE stock list and 4 years of daily prices, and caches them in `data/`. |
| `rules.py` | The rules above, and nothing else. |
| `backtest.py` | Replays the rules over every stock and prints the results. |
| `hindsight.py` | The follow-up: how high each trade got, and 15 alternative exits. |

```bash
cd stocks/pet
python backtest.py            # uses the cache; the first run downloads (~10 min)
python backtest.py --refresh  # downloads fresh prices
```
