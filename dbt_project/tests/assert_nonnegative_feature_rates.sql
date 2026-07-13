select user_id
from {{ ref('mart_user_features') }}
where reward_claim_rate < 0
   or ad_completion_rate < 0
   or instant_claim_rate < 0
