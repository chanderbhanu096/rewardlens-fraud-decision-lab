select
    cast(install_id as varchar) as install_id,
    cast(user_id as varchar) as user_id,
    cast(device_id as varchar) as device_id,
    cast(installed_at as timestamp) as installed_at,
    cast(app_id as varchar) as app_id,
    cast(country as varchar) as country,
    cast(publisher_id as varchar) as publisher_id,
    cast(campaign_id as varchar) as campaign_id,
    cast(attribution_cost_usd as double) as attribution_cost_usd
from {{ source('raw', 'installs') }}

