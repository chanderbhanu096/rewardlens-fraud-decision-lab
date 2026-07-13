select
    cast(device_id as varchar) as device_id,
    cast(os_name as varchar) as os_name,
    cast(os_version as varchar) as os_version,
    cast(is_emulator as boolean) as is_emulator,
    cast(is_rooted as boolean) as is_rooted,
    cast(first_seen_at as timestamp) as first_seen_at
from {{ source('raw', 'devices') }}

