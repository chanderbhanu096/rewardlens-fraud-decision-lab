select
    cast(user_id as varchar) as user_id,
    cast(device_id as varchar) as device_id,
    cast(is_primary as boolean) as is_primary
from {{ source('raw', 'user_devices') }}

