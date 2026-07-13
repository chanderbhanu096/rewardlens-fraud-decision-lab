select r.reward_claim_id
from {{ ref('stg_reward_claims') }} r
join {{ ref('stg_ad_events') }} a using (ad_event_id)
where r.claimed_at < a.viewed_at
