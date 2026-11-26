# Model Improvements

Draft, 2026-10-26. Ideas for raising model quality, and a proposal for what
"acceptable" should mean. Every number here comes from the scripts in
`notebooks/experiments/` and can be reproduced.

**The headline:** the biggest available gain is a one-line configuration fix,
not a new technique. The shipped classifier stops training after about 11
rounds; adding L2 regularization roughly doubles its performance on unseen
data.

---

## Part 1 — What "acceptable" should mean

### Accuracy is the wrong measure

Fraud is 0.17% of transactions. A model that never flags anything is 99.83%
accurate and catches nothing. Every useful measure here is about the flagged
transactions and the fraud:

| Measure | Question it answers |
|---|---|
| **Precision** | Of the transactions we flag, how many are fraud? Sets reviewer workload and customer friction. |
| **Recall** | Of all fraud, how much do we flag? |
| **Dollar recall** | Of all fraud *dollars*, how much do we flag? Closer to the actual loss than recall. |
| **Alerts per day** | Does the flag volume fit the review team? |
| **False-positive rate** | How many legitimate customers get flagged? |

There is no universal "acceptable" number. It falls out of three
business inputs (see *Inputs needed* below): what a review costs, how many
reviews the team can do per day, and how much friction legitimate customers
should tolerate.

### What the improved model achieves

Regularized model (see idea 1), trained on day 1 and scored on day 2: 140,023
transactions, 211 frauds worth $26,889. This is the most realistic test the
dataset allows, because the model has never seen any day-2 transactions.

| Alert rate | Alerts/day | Frauds caught | Precision | Recall | Dollar recall | Legit customers flagged |
|---|---|---|---|---|---|---|
| 0.05% | 70 | 67 | 0.96 | 0.32 | 0.39 | 1 in 46,600 |
| 0.10% | 140 | 131 | 0.94 | 0.62 | 0.52 | 1 in 15,500 |
| **0.15%** | **210** | **163** | **0.78** | **0.77** | **0.63** | **1 in 3,000** |
| 0.20% | 280 | 168 | 0.60 | 0.80 | 0.64 | 1 in 1,250 |
| 0.50% (current) | 700 | 181 | 0.26 | 0.86 | 0.68 | 1 in 270 |
| 1.00% | 1,400 | 184 | 0.13 | 0.87 | 0.73 | 1 in 115 |

The curve bends sharply around 0.15%, which is roughly the fraud rate itself.
Going from 0.15% to the current 0.5% catches 18 more frauds a day at the cost
of 472 more false alarms.

### Cost framing

Daily cost = fraud dollars missed + (alerts × cost per review). With no
model, the whole $26,889 is lost.

| Review cost | Best fixed alert rate | Daily cost | Expected-loss rule | Daily cost |
|---|---|---|---|---|
| $2 | 0.51% | $9,253 | 516 alerts, 75% of $ caught | **$7,748** |
| $10 | 0.15% | $11,809 | 159 alerts, 67% of $ caught | **$10,568** |
| $25 | 0.15% | $15,034 | 103 alerts, 62% of $ caught | **$12,689** |

Two things follow:

- **The right alert rate depends on review cost.** Cheap reviews justify a
  wider net (0.5%); expensive ones a narrow one (0.15%). A rate picked without
  knowing the review cost is a guess, and the current 0.5% was picked for the
  old Isolation Forest.
- **Ranking by expected loss beats a fixed rate at every review cost**, cutting
  daily cost 10–16% (idea 3).

### Proposed acceptance criteria (draft)

Starting point until the business inputs exist, based on what the improved
model achieves on unseen data with room for day-to-day variation:

| Criterion | Proposed bar | Improved model | Shipped model |
|---|---|---|---|
| Average precision, time-forward | ≥ 0.70 | 0.78 | 0.35 |
| Precision at operating point | ≥ 0.70 | 0.78 | 0.19 |
| Recall at operating point | ≥ 0.70 | 0.77 | 0.63 |
| Dollar recall at operating point | ≥ 0.60, aim for 0.80 | 0.63 | not measured |
| Legit customers flagged | ≤ 1 in 2,000 | 1 in 3,000 | 1 in 250 |

The shipped model fails four of five. The improved model passes all five, but
only narrowly on dollar recall, which is the weakest area (idea 6).

### Inputs needed from the business

1. **Cost of one manual review.** Analyst time, plus any customer contact.
2. **Review capacity.** How many alerts per day the team can actually work.
3. **What a flag does.** Hold for review, block outright, or step-up
   verification (for example, a text to the cardholder). This changes what a
   false positive costs.
4. **Tolerance for customer friction.** How many legitimate customers may be
   inconvenienced, and what that costs in churn and support load.
5. **Who bears the loss.** Whether a missed fraud costs the full amount
   (issuer liability) or less (chargeback to merchant).

---

## Part 2 — Ideas, ranked

### Tier 1 — measured, ready to implement

**1. Fix the classifier configuration.** *Biggest win, one-line change.*
The shipped `HistGradientBoostingClassifier` stops after about 11 of its 300
rounds (`n_iter_` = 11, 11, 14, 11, 11 across folds). Its early-stopping check
fires almost immediately, likely because without L2 regularization a few
trees fit the rare fraud rows with extreme values and the validation loss
jumps. Turning early stopping off doesn't help either; the model becomes
erratic (0.554 ± 0.148). L2 regularization fixes it:

| Configuration | Cross-validation AP | Time-forward AP | Rounds trained |
|---|---|---|---|
| Shipped | 0.694 ± 0.042 | 0.349 | 11 |
| `l2_regularization=1.0` only | 0.839 ± 0.027 | 0.777 | 95 |
| Full regularized set | 0.846 ± 0.018 | 0.782 | 138–313 |

The full set is `max_iter=500, learning_rate=0.03, max_leaf_nodes=15,
min_samples_leaf=40, l2_regularization=1.0`; L2 accounts for nearly all of the
gain.

This also corrects an earlier conclusion. The big gap between
cross-validation and the day-2 test (0.69 vs 0.35) was read as fraud patterns
shifting between days. With the configuration fixed the gap mostly closes
(0.85 vs 0.78), so it was mainly the configuration, not drift.

**2. Lower the alert rate to about 0.15–0.2%** once idea 1 ships. At 0.5% the
improved model's precision is only 0.26. See the Part 1 table, and settle
it with the review-cost input.

**3. Decide by expected loss, not a fixed rate.** Flag when
`P(fraud) × amount > review cost`. It focuses reviews where the money is and
cut daily cost 10–16% in every scenario above. It needs probabilities that
mean what they say. The improved model's do: it predicted 230 frauds on day 2
against 211 actual. Trade-off: it catches fewer frauds by count, because it
lets small ones through deliberately.

**4. Things tested that did not help:**
- *Class weighting* (`class_weight="balanced"`): helps the broken shipped
  configuration (0.748), but hurts once regularized (0.756 vs 0.846). Drop it.
- *The Isolation Forest feature:* no measurable gain in any configuration
  tested (regularized: 0.846 hybrid vs 0.848 supervised only). It roughly
  doubles training and scoring cost. Worth deciding whether the hedge against
  unseen fraud patterns is worth keeping.
- *Hour-of-day feature:* +0.01 cross-validation, +0.02 time-forward, within
  noise. Two days of data are too short to learn a daily pattern; revisit
  with more history.

### Tier 2 — promising, not yet measured

**5. Proper hyperparameter search.** Idea 1 is one hand-picked configuration.
A search scored on the time-forward split (not just cross-validation) should
add a few points on top.

**6. Find out why large frauds are missed.** Dollar recall trails count recall
throughout (0.63 vs 0.77 at 0.15%), so the model misses disproportionately
large frauds. A segment analysis of missed frauds by amount should show
whether this is fixable or a limitation of the features.

**7. Tiered actions.** Precision is 0.96 at the top 0.05%, which is high enough
to block outright. A three-tier policy (block the top, review the middle, pass
the rest) would cut review volume without losing catches.

**8. Try other model families as a cross-check.** Plain logistic regression
scored 0.759 / 0.660, beating the shipped model, which is a useful sanity
baseline. LightGBM or XGBoost are unlikely to beat a tuned gradient-boosted
model by much, but averaging a linear and a tree model sometimes helps.

### Tier 3 — needs data this dataset does not have

These are where real-world fraud systems get most of their accuracy, and none
are possible with the Kaggle data, which is anonymized PCA components with no
card, merchant, or customer identifiers.

**9. Behavioral and velocity features.** Transactions per card in the last
hour, amount compared with the card's usual spend, a merchant never seen for
this card, distance from the last transaction. Requires card and merchant IDs
in the `/ingest` contract and a feature store the scorer can read.

**10. A label feedback loop.** Reviewed alerts become new training labels.
Two traps: only flagged transactions get reviewed, so labels are biased toward
what the model already catches; and chargebacks confirm fraud weeks to
months later, so the newest data is under-labelled.

**11. Monitoring and retraining.** Track score distribution, daily alert rate,
and the precision of reviewed alerts. Retrain on a schedule, and when precision
drops. The two-day dataset can't tell us how fast real fraud drifts, so set
the cadence from production evidence.

---

## Caveats

- **Small samples.** Day 2 has 211 frauds; every precision and recall figure
  could move several points on a different day.
- **Costs assume a flag prevents the loss.** The pipeline scores
  asynchronously, after the transaction is accepted. Preventing loss needs a
  hold before settlement, or a synchronous scoring path for high-risk
  transactions.
- **Review costs are placeholders** until the business inputs exist.
- **The numbers won't transfer as-is.** Production features will differ from
  Kaggle's PCA components; treat these as evidence for the approach, not
  promises.

## Suggested order

1. Ship idea 1 (L2 fix) and retune the alert rate (idea 2). Small change,
   largest gain.
2. Collect the business inputs and finalize the acceptance criteria.
3. Implement the expected-loss rule (idea 3) and tiered actions (idea 7).
4. Hyperparameter search and the large-fraud analysis (ideas 5, 6).
5. Plan the data work for Tier 3, which is where the long-term gains are.
