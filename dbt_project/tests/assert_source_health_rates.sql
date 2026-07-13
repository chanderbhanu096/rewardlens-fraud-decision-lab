select *
from {{ ref('mart_source_daily_health') }}
where instant_claim_rate not between 0 and 1
   or prior_observed_days is null
   or prior_observed_days not between 0 and 7
   or (
       prior_observed_days < 7
       and (
           prior_7d_instant_claim_rate is not null
           or instant_claim_rate_lift_pp is not null
       )
   )
   or (
       prior_observed_days = 7
       and (
           prior_7d_instant_claim_rate is null
           or instant_claim_rate_lift_pp is null
       )
   )
   or (
       prior_7d_instant_claim_rate is not null
       and prior_7d_instant_claim_rate not between 0 and 1
   )
