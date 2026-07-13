select
    country,
    publisher_id,
    count(*) as users,
    avg(sessions_per_active_day) as avg_sessions_per_active_day,
    stddev_pop(sessions_per_active_day) as sd_sessions_per_active_day,
    avg(reward_claim_rate) as avg_reward_claim_rate,
    stddev_pop(reward_claim_rate) as sd_reward_claim_rate,
    avg(median_claim_delay_seconds) as avg_claim_delay_seconds,
    stddev_pop(median_claim_delay_seconds) as sd_claim_delay_seconds,
    avg(net_reward_loss_usd) as avg_net_reward_loss_usd
from {{ ref('mart_user_features') }}
group by 1, 2
