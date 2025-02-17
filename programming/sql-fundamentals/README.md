# SQL Fundamentals for Machine Learning

This section covers essential SQL concepts and best practices for working with large datasets in machine learning projects.

## Basic SQL Concepts

### 1. Query Fundamentals
- SELECT, FROM, WHERE
- GROUP BY, HAVING
- ORDER BY, LIMIT
- JOIN operations
- Subqueries
- Common Table Expressions (CTEs)

### 2. Data Types
- Numeric types
- Character types
- Date and time types
- Boolean type
- Arrays and JSON
- Binary types

### 3. Database Design
- Primary and foreign keys
- Indexes
- Normalization
- Constraints
- Views
- Materialized views

## Advanced SQL

### 1. Window Functions
```sql
-- Running averages
SELECT 
    date,
    value,
    AVG(value) OVER (
        ORDER BY date 
        ROWS BETWEEN 7 PRECEDING AND CURRENT ROW
    ) as moving_avg
FROM time_series;

-- Rank and dense_rank
SELECT 
    category,
    value,
    RANK() OVER (PARTITION BY category ORDER BY value DESC) as rank,
    DENSE_RANK() OVER (PARTITION BY category ORDER BY value DESC) as dense_rank
FROM measurements;
```

### 2. Advanced Aggregations
```sql
-- Conditional aggregation
SELECT 
    category,
    COUNT(*) as total_count,
    COUNT(CASE WHEN value > 100 THEN 1 END) as high_value_count,
    AVG(CASE WHEN date >= '2024-01-01' THEN value END) as recent_avg
FROM data
GROUP BY category;

-- Pivot operations
SELECT *
FROM CROSSTAB(
    'SELECT category, year, COUNT(*)
     FROM events
     GROUP BY category, year
     ORDER BY 1,2'
) AS ct(category text, "2022" int, "2023" int, "2024" int);
```

## Performance Optimization

### 1. Index Optimization
```sql
-- Create indexes
CREATE INDEX idx_timestamp ON measurements(timestamp);
CREATE INDEX idx_category_value ON measurements(category, value);

-- Partial indexes
CREATE INDEX idx_high_value ON measurements(value)
WHERE value > 1000;
```

### 2. Query Optimization
```sql
-- Use EXISTS instead of IN for better performance
SELECT *
FROM customers c
WHERE EXISTS (
    SELECT 1 
    FROM orders o 
    WHERE o.customer_id = c.id
);

-- Optimize JOIN operations
SELECT /*+ HASH_JOIN */ c.name, o.order_date
FROM customers c
JOIN orders o ON c.id = o.customer_id;
```

## Working with Large Datasets

### 1. Batch Processing
```sql
-- Process data in batches
WITH batch AS (
    SELECT id
    FROM large_table
    WHERE processed = false
    ORDER BY id
    LIMIT 1000
    FOR UPDATE SKIP LOCKED
)
UPDATE large_table
SET processed = true
WHERE id IN (SELECT id FROM batch);
```

### 2. Partitioning
```sql
-- Create partitioned table
CREATE TABLE measurements (
    id SERIAL,
    timestamp TIMESTAMP,
    value NUMERIC
) PARTITION BY RANGE (timestamp);

-- Create partitions
CREATE TABLE measurements_2024 PARTITION OF measurements
    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');
```

## Machine Learning Applications

### 1. Feature Engineering
```sql
-- Time-based features
SELECT 
    id,
    EXTRACT(HOUR FROM timestamp) as hour_of_day,
    EXTRACT(DOW FROM timestamp) as day_of_week,
    value,
    value - LAG(value) OVER (ORDER BY timestamp) as value_change
FROM time_series;

-- Statistical features
SELECT 
    category,
    AVG(value) as mean,
    STDDEV(value) as std_dev,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY value) as median
FROM measurements
GROUP BY category;
```

### 2. Data Preparation
```sql
-- Handle missing values
SELECT 
    id,
    COALESCE(value, AVG(value) OVER ()) as imputed_value,
    CASE WHEN value IS NULL THEN 1 ELSE 0 END as is_imputed
FROM measurements;

-- Normalize features
WITH stats AS (
    SELECT 
        AVG(value) as mean,
        STDDEV(value) as std
    FROM measurements
)
SELECT 
    id,
    (value - mean) / NULLIF(std, 0) as normalized_value
FROM measurements, stats;
```

## Best Practices

### 1. Query Writing
- Use meaningful aliases
- Format SQL for readability
- Comment complex queries
- Use CTEs for clarity
- Avoid SELECT *

### 2. Performance
- Use appropriate indexes
- Monitor query execution plans
- Optimize JOIN operations
- Use appropriate data types
- Regular maintenance (VACUUM, ANALYZE)

## Resources

### Books
- "SQL Performance Explained" by Markus Winand
- "SQL Antipatterns" by Bill Karwin
- "Learning SQL" by Alan Beaulieu

### Online Resources
- Mode Analytics SQL Tutorial
- PostgreSQL Documentation
- SQLZoo Interactive Tutorials

### Tools
- DBeaver
- pgAdmin
- SQL Server Management Studio
- EXPLAIN ANALYZE

## Assessment Questions

1. How do indexes improve query performance?
2. Explain the difference between RANK and DENSE_RANK
3. When should you use partitioning?
4. How can you optimize JOIN operations?

## Projects

1. Design a database for ML feature storage
2. Create an ETL pipeline
3. Implement efficient batch processing
4. Build a feature engineering framework
