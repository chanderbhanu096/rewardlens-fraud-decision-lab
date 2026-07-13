select *
from {{ ref('stg_sessions') }}
where ended_at < started_at

