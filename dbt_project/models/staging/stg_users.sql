select
    cast(user_id as varchar) as user_id,
    cast(country as varchar) as country,
    cast(created_at as timestamp) as created_at,
    cast(fraud_type as varchar) as fraud_type,
    cast(is_fraud as boolean) as is_fraud
from {{ source('raw', 'users') }}

