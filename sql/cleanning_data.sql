SELECT DISTINCT 
    person_age, 
    person_income, 
    person_emp_length, 
    loan_amnt, 
    loan_int_rate, 
    loan_status
FROM db_credit_risk
WHERE (person_age <= 100)
  AND (person_emp_length <= person_age OR person_emp_length IS NULL)
  AND (loan_amnt > 0)
  AND (person_income > 0);