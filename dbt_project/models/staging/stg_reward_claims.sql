select
    cast(reward_claim_id as varchar) as reward_claim_id,
    cast(ad_event_id as varchar) as ad_event_id,
    cast(user_id as varchar) as user_id,
    cast(device_id as varchar) as device_id,
    cast(claimed_at as timestamp) as claimed_at,
    cast(seconds_after_ad as double) as seconds_after_ad,
    cast(reward_type as varchar) as reward_type,
    cast(reward_value_usd as double) as reward_value_usd
from {{ source('raw', 'reward_claims') }}

