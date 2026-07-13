# Business recommendation

RewardLens should proceed with a targeted, reversible pilot of the balanced fraud
threshold—not an automatic global block. The balanced policy maximizes modeled
value, but the simulated retention interval fails the pre-specified
-2 percentage-point non-inferiority guardrail.

Recommended operating policy:

1. Automatically hold rewards only for top-5% risk-rank cases during the pilot.
2. Route users above the balanced cutoff to review or step-up verification.
3. Monitor publisher and country cohorts for sudden shifts.
4. Recalibrate economic assumptions and thresholds weekly.
5. Do not expand until a confirmatory test clears the pre-registered retention
   non-inferiority margin.

The generated experiment intentionally contains heterogeneous country effects and
an uncertain overall retention result. Its executable recommendation is written to
`artifacts/experiment/recommendation.md` after every pipeline run.
