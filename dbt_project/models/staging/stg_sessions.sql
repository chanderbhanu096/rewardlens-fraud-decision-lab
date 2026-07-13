select
    cast(session_id as varchar) as session_id,
    cast(user_id as varchar) as user_id,
    cast(device_id as varchar) as device_id,
    cast(started_at as timestamp) as started_at,
    cast(ended_at as timestamp) as ended_at,
    cast(duration_seconds as integer) as duration_seconds,
    cast(level_gain as integer) as level_gain
from {{ source('raw', 'sessions') }}

