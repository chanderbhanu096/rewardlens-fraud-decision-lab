with session_daily as (
    select
        user_id,
        cast(started_at as date) as activity_date,
        count(*) as session_count,
        sum(duration_seconds) as session_seconds,
        sum(level_gain) as level_gain
    from {{ ref('stg_sessions') }}
    group by 1, 2
),
ad_daily as (
    select
        user_id,
        cast(viewed_at as date) as activity_date,
        count(*) as ad_views,
        sum(case when completed then 1 else 0 end) as completed_ads,
        sum(revenue_usd) as ad_revenue_usd
    from {{ ref('stg_ad_events') }}
    group by 1, 2
),
reward_daily as (
    select
        user_id,
        cast(claimed_at as date) as activity_date,
        count(*) as reward_claims,
        median(seconds_after_ad) as median_claim_delay_seconds,
        sum(reward_value_usd) as reward_cost_usd
    from {{ ref('stg_reward_claims') }}
    group by 1, 2
),
spine as (
    select user_id, activity_date from session_daily
    union
    select user_id, activity_date from ad_daily
    union
    select user_id, activity_date from reward_daily
),
joined as (
    select
        spine.user_id,
        spine.activity_date,
        coalesce(s.session_count, 0) as session_count,
        coalesce(s.session_seconds, 0) as session_seconds,
        coalesce(s.level_gain, 0) as level_gain,
        coalesce(a.ad_views, 0) as ad_views,
        coalesce(a.completed_ads, 0) as completed_ads,
        coalesce(a.ad_revenue_usd, 0) as ad_revenue_usd,
        coalesce(r.reward_claims, 0) as reward_claims,
        r.median_claim_delay_seconds,
        coalesce(r.reward_cost_usd, 0) as reward_cost_usd
    from spine
    left join session_daily s using (user_id, activity_date)
    left join ad_daily a using (user_id, activity_date)
    left join reward_daily r using (user_id, activity_date)
)
select
    *,
    sum(session_count) over user_window as sessions_7d,
    sum(ad_views) over user_window as ad_views_7d,
    sum(reward_claims) over user_window as reward_claims_7d,
    sum(reward_cost_usd) over user_window as reward_cost_7d_usd
from joined
window user_window as (
    partition by user_id
    order by activity_date
    range between interval 6 day preceding and current row
)

