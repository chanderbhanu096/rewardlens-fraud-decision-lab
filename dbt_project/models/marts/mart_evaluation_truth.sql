-- Isolated from the feature mart to prevent accidental model leakage.
select user_id, fraud_type, is_fraud
from {{ ref('stg_users') }}

