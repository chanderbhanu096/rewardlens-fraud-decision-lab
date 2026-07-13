select
    f.*,
    (f.sessions_per_active_day - b.avg_sessions_per_active_day)
        / nullif(b.sd_sessions_per_active_day, 0) as sessions_peer_zscore,
    (f.reward_claim_rate - b.avg_reward_claim_rate)
        / nullif(b.sd_reward_claim_rate, 0) as claim_rate_peer_zscore,
    (f.median_claim_delay_seconds - b.avg_claim_delay_seconds)
        / nullif(b.sd_claim_delay_seconds, 0) as claim_delay_peer_zscore
from {{ ref('mart_user_features') }} f
join {{ ref('mart_population_baselines') }} b
    using (country, publisher_id)
