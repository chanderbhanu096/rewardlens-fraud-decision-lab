select
    activity_date,
    source_type,
    source_id,
    count(*) as row_count
from {{ ref('mart_source_daily_health') }}
group by 1, 2, 3
having count(*) > 1
