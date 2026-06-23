BEGIN;

-- Invalid age -> NULL
UPDATE db_credit_risk
SET person_age = NULL
WHERE person_age >= 100;

-- Fill NULL with median
WITH median_age AS (
    SELECT PERCENTILE_CONT(0.5)
           WITHIN GROUP (ORDER BY person_age) AS median
    FROM db_credit_risk
    WHERE person_age IS NOT NULL
)
UPDATE db_credit_risk
SET person_age = median_age.median
FROM median_age
WHERE db_credit_risk.person_age IS NULL;

-- Fill employment length
UPDATE db_credit_risk
SET person_emp_length = 1
WHERE person_emp_length IS NULL;

COMMIT;