# RewardLens synthetic data contract

All timestamps are UTC. Monetary values are USD. IDs are stable only within a
generated dataset. `fraud_type` and `is_fraud` are evaluation-only ground truth
and must never be used as model inputs.

## Tables

| Table | Grain | Important columns |
|---|---|---|
| `users` | One row per account | `user_id`, `country`, `created_at`, `fraud_type`, `is_fraud` |
| `devices` | One row per physical/virtual device | OS, emulator/root indicators, first seen time |
| `user_devices` | One row per user-device link | Enables many accounts on one device |
| `installs` | One attributed install per user | App, publisher, campaign, country, cost |
| `sessions` | One gameplay session | Start/end, duration, level progress |
| `ad_events` | One rewarded-ad view | Network, placement, completion, revenue |
| `reward_claims` | One claimed reward | Claim delay, reward type, value |
| `experiment_assignments` | One assignment per user | 50/50 country-stratified control/treatment |

## Planted populations

| Population | Main signals | Intended ambiguity |
|---|---|---|
| Normal | Diverse timing, gameplay progress, modest ad usage | Occasional shared/rooted device |
| Reward abuse | Short claim delay, many rewarded ads, short sessions | Mostly non-emulated devices |
| Emulator farm | Many accounts per device, synchronized timing, low progress | Some emulators evade the flag |
| Click farm | Repetitive/high-frequency short sessions, concentrated publishers | Lower reward-claim rate |

The planted labels are correlated with suspicious signals but not identical to
them. This supports realistic false-positive and false-negative trade-offs.
