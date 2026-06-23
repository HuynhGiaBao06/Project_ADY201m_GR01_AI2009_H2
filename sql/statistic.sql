SELECT 
    CASE 
        WHEN person_age <= 25 THEN 'Under 25'
        WHEN person_age >= 26 AND person_age < 40 THEN '26 to 39'
        WHEN person_age >= 40 THEN '40 and above'
    END AS age_group,
    AVG(person_emp_length) AS avg_emp_length, -- Tính trung bình thời gian làm việc
    COUNT(*) AS total_people                  -- Đếm số lượng người trong mỗi nhóm
FROM db_credit_risk
GROUP BY 
    CASE 
        WHEN person_age <= 25 THEN 'Under 25'
        WHEN person_age >= 26 AND person_age < 40 THEN '26 to 39'
        WHEN person_age >= 40 THEN '40 and above'
    END;