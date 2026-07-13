with observation as (
    select max(started_at) as observation_at from {{ ref('stg_sessions') }}
),
session_features as (
    select
        user_id,
        count(*) as session_count,
        count(distinct cast(started_at as date)) as active_days,
        avg(duration_seconds) as avg_session_seconds,
        stddev_pop(duration_seconds) as stddev_session_seconds,
        sum(level_gain) as total_level_gain,
        avg(level_gain) as avg_level_gain,
        count(distinct date_part('hour', started_at)) as distinct_session_hours
    from {{ ref('stg_sessions') }}
    group by 1
),
ad_features as (
    select
        user_id,
        count(*) as ad_views,
        sum(case when completed then 1 else 0 end) as completed_ads,
        sum(revenue_usd) as ad_revenue_usd,
        count(distinct ad_network) as ad_networks
    from {{ ref('stg_ad_events') }}
    group by 1
),
reward_features as (
    select
        user_id,
        count(*) as reward_claims,
        median(seconds_after_ad) as median_claim_delay_seconds,
        quantile_cont(seconds_after_ad, 0.1) as p10_claim_delay_seconds,
        sum(case when seconds_after_ad < 3 then 1 else 0 end) as instant_claims,
        sum(reward_value_usd) as reward_cost_usd
    from {{ ref('stg_reward_claims') }}
    group by 1
)
select
    u.user_id,
    i.device_id,
    i.app_id,
    i.country,
    i.publisher_id,
    i.campaign_id,
    i.installed_at,
    i.attribution_cost_usd,
    date_diff('day', i.installed_at, o.observation_at) as account_age_days,
    d.is_emulator,
    d.is_rooted,
    da.users_on_device,
    da.account_creation_span_hours,
    coalesce(s.session_count, 0) as session_count,
    coalesce(s.active_days, 0) as active_days,
    coalesce(s.session_count / nullif(s.active_days, 0), 0) as sessions_per_active_day,
    coalesce(s.avg_session_seconds, 0) as avg_session_seconds,
    coalesce(s.stddev_session_seconds, 0) as stddev_session_seconds,
    coalesce(s.total_level_gain, 0) as total_level_gain,
    coalesce(s.avg_level_gain, 0) as avg_level_gain,
    coalesce(s.distinct_session_hours, 0) as distinct_session_hours,
    coalesce(a.ad_views, 0) as ad_views,
    coalesce(a.completed_ads, 0) as completed_ads,
    coalesce(a.completed_ads / nullif(a.ad_views, 0), 0) as ad_completion_rate,
    coalesce(a.ad_revenue_usd, 0) as ad_revenue_usd,
    coalesce(a.ad_networks, 0) as ad_networks,
    coalesce(r.reward_claims, 0) as reward_claims,
    coalesce(r.reward_claims / nullif(a.completed_ads, 0), 0) as reward_claim_rate,
    coalesce(r.median_claim_delay_seconds, 999) as median_claim_delay_seconds,
    coalesce(r.p10_claim_delay_seconds, 999) as p10_claim_delay_seconds,
    coalesce(r.instant_claims, 0) as instant_claims,
    coalesce(r.instant_claims / nullif(r.reward_claims, 0), 0) as instant_claim_rate,
    coalesce(r.reward_cost_usd, 0) as reward_cost_usd,
    coalesce(r.reward_cost_usd, 0) - coalesce(a.ad_revenue_usd, 0) as net_reward_loss_usd
from {{ ref('stg_users') }} u
join {{ ref('stg_installs') }} i using (user_id)
join {{ ref('stg_devices') }} d using (device_id)
join {{ ref('int_device_accounts') }} da using (device_id)
cross join observation o
left join session_features s using (user_id)
left join ad_features a using (user_id)
left join reward_features r using (user_id)
