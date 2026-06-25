WITH MedianByGrade AS (
    SELECT
        loan_grade,
        PERCENTILE_CONT(0.5)
            WITHIN GROUP (ORDER BY loan_int_rate) AS median
    FROM db_credit_risk
    WHERE loan_int_rate IS NOT NULL
    GROUP BY loan_grade
),

CleanData AS (
    SELECT
        d.id,
        d.person_age,
        d.person_income,
        COALESCE(d.person_emp_length, 1) AS person_emp_length,
        d.loan_amnt,
        COALESCE(d.loan_int_rate, m.median) AS loan_int_rate,
        d.loan_status,
        ROW_NUMBER() OVER (PARTITION BY d.id ORDER BY d.id) AS rn
    FROM db_credit_risk d
    LEFT JOIN MedianByGrade m
        ON d.loan_grade = m.loan_grade
    WHERE d.person_age <= 100
      AND (d.person_emp_length <= d.person_age OR d.person_emp_length IS NULL)
      AND d.loan_amnt > 0
      AND d.person_income >= 0
)

SELECT *
FROM CleanData
WHERE rn = 1;