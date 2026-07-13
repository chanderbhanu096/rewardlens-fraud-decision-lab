select
    cast(experiment_id as varchar) as experiment_id,
    cast(user_id as varchar) as user_id,
    cast(variant as varchar) as variant,
    cast(assigned_at as timestamp) as assigned_at
from {{ source('raw', 'experiment_assignments') }}

