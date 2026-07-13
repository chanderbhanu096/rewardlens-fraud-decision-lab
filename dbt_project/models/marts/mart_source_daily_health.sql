with claim_events as (
    select
        cast(reward.claimed_at as date) as activity_date,
        install.publisher_id,
        install.campaign_id,
        reward.user_id,
        reward.seconds_after_ad,
        reward.reward_value_usd
    from {{ ref('stg_reward_claims') }} as reward
    inner join {{ ref('stg_installs') }} as install using (user_id)
),

source_events as (
    select
        activity_date,
        'publisher' as source_type,
        publisher_id as source_id,
        user_id,
        seconds_after_ad,
        reward_value_usd
    from claim_events

    union all

    select
        activity_date,
        'campaign' as source_type,
        campaign_id as source_id,
        user_id,
        seconds_after_ad,
        reward_value_usd
    from claim_events
),

daily as (
    select
        activity_date,
        source_type,
        source_id,
        count(*) as reward_claims,
        count(distinct user_id) as users_claiming,
        sum(case when seconds_after_ad < 3 then 1 else 0 end) as instant_claims,
        sum(reward_value_usd) as reward_cost_usd
    from source_events
    group by 1, 2, 3
),

with_prior_window as (
    select
        *,
        count(*) over (
            partition by source_type, source_id
            order by activity_date
            rows between 7 preceding and 1 preceding
        ) as prior_observed_days,
        sum(instant_claims) over (
            partition by source_type, source_id
            order by activity_date
            rows between 7 preceding and 1 preceding
        ) as prior_7d_instant_claims,
        sum(reward_claims) over (
            partition by source_type, source_id
            order by activity_date
            rows between 7 preceding and 1 preceding
        ) as prior_7d_reward_claims
    from daily
)

select
    activity_date,
    source_type,
    source_id,
    reward_claims,
    users_claiming,
    instant_claims,
    instant_claims::double / nullif(reward_claims, 0) as instant_claim_rate,
    reward_cost_usd,
    prior_observed_days,
    case
        when prior_observed_days = 7
            then prior_7d_instant_claims::double
                / nullif(prior_7d_reward_claims, 0)
    end as prior_7d_instant_claim_rate,
    case
        when prior_observed_days = 7
            then 100 * (
                instant_claims::double / nullif(reward_claims, 0)
                - prior_7d_instant_claims::double
                    / nullif(prior_7d_reward_claims, 0)
            )
    end as instant_claim_rate_lift_pp
from with_prior_window
