# RewardLens learning guide

This guide explains the same system at increasing levels of depth. Stop when the
idea is clear; continue when you want to understand the implementation.

## Level 1 — the story

CoinQuest gives a player 100 coins after they watch an advertisement. Maya plays
normally: one account, one phone, several minutes of gameplay, then an ad and a
reward. A fraud operator uses virtual phones to create hundreds of accounts and
repeat the same reward sequence every few seconds.

RewardLens helps CoinQuest answer one question:

> Which rewards need review without punishing valuable players like Maya?

It is a decision-support system. It does not declare that every unusual player is
a fraudster.

## Level 2 — the data journey

One recorded action is an **event**. Examples include an install, session start,
ad view, or reward claim. The system summarizes many events into **features**:

| Raw behaviour | Feature used as a clue |
|---|---|
| 300 accounts touch one virtual device | `users_on_device = 300` |
| A reward is claimed one second after an ad | `median_claim_delay_seconds = 1` |
| Sessions repeat throughout the day | `sessions_per_active_day = 240` |
| Accounts perform almost no gameplay | `avg_level_gain = 0.05` |

The complete flow is:

```text
events → validated SQL models → behavioural features → anomaly detectors
       → risk rank → intervention policy → experiment decision
```

The important distinction is between a **score** and an **action**. A score only
orders cases for attention. A policy decides whether to release a reward, request
verification, send a case to review, or temporarily hold it.

## Level 3 — how the score is built

Four detectors look for different failure modes:

1. **Weighted rules** encode known warning signs such as device sharing and
   instant claims.
2. **Robust deviations** compare feature values with the cohort median using the
   median absolute deviation. This is less sensitive to extreme values than a
   mean and standard deviation.
3. **Isolation Forest** finds uncommon combinations of features.
4. **DBSCAN rarity** detects accounts in sparse behavioural regions or noise.

Their cohort-relative outputs are combined explicitly:

```text
combined score
= 0.34 × rules
+ 0.24 × robust deviation
+ 0.32 × isolation score
+ 0.10 × cluster rarity
```

The weights are documented design assumptions. They were not fitted using the
planted fraud labels.

Each scored row contains:

- the four detector scores;
- the weighted contribution from each detector;
- the dominant detector;
- the largest robust feature deviation;
- triggered rule explanations;
- a within-run risk rank and operational tier.

A risk rank of 97% means the account ranks above approximately 97% of this
scoring batch. It is not a 97% probability of fraud.

## Level 4 — the decision science

Offline evaluation joins the score to planted synthetic truth only after scoring.
The confusion matrix separates four outcomes:

| Outcome | Meaning |
|---|---|
| True positive | Fraudulent account correctly flagged |
| False positive | Legitimate account incorrectly flagged |
| False negative | Fraudulent account missed |
| True negative | Legitimate account correctly released |

The main measures answer different questions:

- **Precision:** of the accounts flagged, how many were fraudulent?
- **Recall:** of all fraudulent accounts, how many were caught?
- **False-positive rate:** of legitimate accounts, how many were flagged?
- **Alerts per 1,000:** how much operational review capacity is required?
- **Net savings:** how much projected fraud loss remains after modeled customer
  friction?

In the reference run, Balanced catches the same 900 planted fraudulent accounts
as Aggressive. Aggressive additionally flags 800 legitimate users. More alerts
therefore create less modeled value.

The economic result depends on assumptions. RewardLens stress-tests every policy
across 16 combinations of fraud-loss horizon and false-positive cost. This makes
recommendation instability visible instead of hiding it behind one dollar value.

## Level 5 — experiment interpretation

Good offline performance is not permission to launch. The simulated A/B test
randomly assigns users within country to:

- **control:** current fraud policy;
- **treatment:** the new Balanced policy.

Fraud reward cost per assigned user is the primary intent-to-treat outcome.
Day-7 retention is a pre-specified non-inferiority guardrail with a −2 percentage
point margin.

The treatment reduces simulated fraud cost by 77%, but the retention confidence
interval extends below the allowed margin. A p-value of 0.051 does not prove
safety; “not statistically significant” is not the same as “no harmful effect.”
The correct recommendation is therefore a small, reversible targeted pilot.

Country estimates are exploratory. Benjamini–Hochberg adjustment controls the
false-discovery rate across the segment comparisons. Positive-looking segments
are hypotheses for the next test, not evidence for immediate rollout.

## Level 6 — implementation review

| Layer | Review question | Location |
|---|---|---|
| Event contract | Are table grains, keys, and synthetic populations explicit? | `data_generator/SCHEMA.md` |
| Generator | Are results deterministic and relationally consistent? | `data_generator/generate.py` |
| Transformation | Are features reusable and tested? | `dbt_project/models/` |
| Leakage boundary | Can truth enter scoring before evaluation? | `mart_evaluation_truth.sql`, `validate_scoring_frame` |
| Scoring | Are assumptions and per-user contributions inspectable? | `anomaly_detection/scoring.py` |
| Experiment | Are primary, guardrail, multiplicity, and decision gates explicit? | `experiment_analysis/analyze.py` |
| Product | Can an analyst move from portfolio risk to one explainable case? | `dashboard/app.py` |
| Reproducibility | Can one command rebuild the complete result? | `orchestration/pipeline.py` |

The scoring contract fails fast when features are missing, user IDs are not
unique, detector weights do not sum to one, or offline truth appears in the
scoring frame. These checks turn analytical assumptions into executable
guarantees.

## What an expert should challenge

The project deliberately exposes its remaining gaps:

1. Evaluation is in-sample and synthetic; production needs temporal and
   out-of-time validation.
2. Percentile ranking controls queue size but can hide population-wide drift;
   production needs frozen reference distributions and drift monitors.
3. Unsupervised scores are not calibrated probabilities; actions should be tied
   to review capacity and measured costs.
4. Peer comparisons need minimum cohort sizes, shrinkage, and as-of-time joins.
5. Real labels are delayed and selection-biased; feedback from investigation and
   appeals must be modeled explicitly.
6. Financial coefficients need validation by finance, support, product, and
   experimentation owners.

That is the intended senior-level conclusion: RewardLens is not “a model that
finds fraud.” It is an auditable framework for making and testing intervention
decisions under uncertainty.
