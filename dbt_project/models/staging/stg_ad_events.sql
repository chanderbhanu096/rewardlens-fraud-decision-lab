select
    cast(ad_event_id as varchar) as ad_event_id,
    cast(session_id as varchar) as session_id,
    cast(user_id as varchar) as user_id,
    cast(device_id as varchar) as device_id,
    cast(viewed_at as timestamp) as viewed_at,
    cast(ad_network as varchar) as ad_network,
    cast(placement as varchar) as placement,
    cast(completed as boolean) as completed,
    cast(revenue_usd as double) as revenue_usd
from {{ source('raw', 'ad_events') }}

