select
    ud.device_id,
    count(distinct ud.user_id) as users_on_device,
    min(u.created_at) as first_account_at,
    max(u.created_at) as last_account_at,
    date_diff('hour', min(u.created_at), max(u.created_at)) as account_creation_span_hours
from {{ ref('stg_user_devices') }} ud
join {{ ref('stg_users') }} u using (user_id)
group by 1

