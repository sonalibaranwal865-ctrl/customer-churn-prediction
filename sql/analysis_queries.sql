-- SQL practice for a table named customers.
-- Adjust column/table names to match your actual imported dataset.
-- Example MySQL import: create a table with matching columns, then import the CSV using Workbench.

-- 1. Overall customer count
SELECT COUNT(*) AS total_customers
FROM customers;

-- 2. Churn count and rate
SELECT
    Churn,
    COUNT(*) AS customer_count,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM customers), 2) AS percentage
FROM customers
GROUP BY Churn;

-- 3. Churn by contract type
SELECT
    Contract,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers
FROM customers
GROUP BY Contract
ORDER BY churned_customers DESC;

-- 4. Average monthly charges by churn status
SELECT Churn, ROUND(AVG(MonthlyCharges), 2) AS average_monthly_charges
FROM customers
GROUP BY Churn;

-- 5. Churn by tenure band
SELECT
    CASE
        WHEN tenure < 12 THEN '0-11 months'
        WHEN tenure < 24 THEN '12-23 months'
        WHEN tenure < 48 THEN '24-47 months'
        ELSE '48+ months'
    END AS tenure_band,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers
FROM customers
GROUP BY tenure_band
ORDER BY MIN(tenure);
