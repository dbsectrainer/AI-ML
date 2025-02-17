# Big Data Tools for Machine Learning

This section covers essential big data tools and frameworks used in machine learning projects.

## Hadoop Ecosystem

### 1. HDFS (Hadoop Distributed File System)
```bash
# Basic HDFS commands
hdfs dfs -ls /path                  # List files
hdfs dfs -put localfile /path       # Upload file
hdfs dfs -get /path/file localfile  # Download file
hdfs dfs -rm /path/file             # Remove file
hdfs dfs -mkdir /path               # Create directory
```

### 2. MapReduce Programming
```python
from mrjob.job import MRJob

class WordCount(MRJob):
    def mapper(self, _, line):
        for word in line.split():
            yield word.lower(), 1
            
    def reducer(self, word, counts):
        yield word, sum(counts)

if __name__ == '__main__':
    WordCount.run()
```

### 3. Hive for Data Warehousing
```sql
-- Create external table
CREATE EXTERNAL TABLE logs (
    timestamp BIGINT,
    user_id STRING,
    action STRING,
    parameters MAP<STRING, STRING>
)
STORED AS PARQUET
LOCATION '/data/logs';

-- Query with partitioning
SELECT 
    action,
    COUNT(*) as count
FROM logs
WHERE timestamp >= unix_timestamp(current_date - 7)
GROUP BY action;
```

## Apache Spark

### 1. PySpark Basics
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

# Initialize Spark
spark = SparkSession.builder \
    .appName("ML Pipeline") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

# Read and process data
df = spark.read.parquet("/data/features.parquet")
df_processed = df.filter(col("value") > 0) \
    .groupBy("category") \
    .agg(avg("value").alias("avg_value"))
```

### 2. Spark ML Pipeline
```python
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier

# Create feature vector
assembler = VectorAssembler(
    inputCols=["feature1", "feature2", "feature3"],
    outputCol="features"
)

# Scale features
scaler = StandardScaler(
    inputCol="features",
    outputCol="scaled_features"
)

# Create classifier
rf = RandomForestClassifier(
    labelCol="label",
    featuresCol="scaled_features",
    numTrees=100
)

# Build pipeline
pipeline = Pipeline(stages=[assembler, scaler, rf])
model = pipeline.fit(training_data)
```

### 3. Spark Streaming
```python
from pyspark.streaming import StreamingContext

# Create streaming context
ssc = StreamingContext(sc, batchDuration=1)

# Create DStream
lines = ssc.socketTextStream("localhost", 9999)

# Process streaming data
word_counts = lines.flatMap(lambda line: line.split(" ")) \
    .map(lambda word: (word, 1)) \
    .reduceByKey(lambda a, b: a + b)

word_counts.pprint()
ssc.start()
ssc.awaitTermination()
```

## Distributed Computing

### 1. Dask for Parallel Computing
```python
import dask.dataframe as dd
import dask.array as da
from dask.distributed import Client

# Initialize Dask client
client = Client()

# Read and process large dataset
df = dd.read_csv('data/*.csv')
result = df.groupby('category')['value'].mean().compute()

# Parallel array operations
array = da.random.random((10000, 10000), chunks=(1000, 1000))
result = array.mean(axis=0).compute()
```

### 2. Ray for Distributed ML
```python
import ray
from ray import tune

# Initialize Ray
ray.init()

def train_model(config):
    # Model training logic
    accuracy = model.train(**config)
    tune.report(mean_accuracy=accuracy)

# Hyperparameter tuning
analysis = tune.run(
    train_model,
    config={
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([16, 32, 64, 128]),
        "hidden_units": tune.choice([64, 128, 256])
    },
    num_samples=50
)
```

## Data Lake Architecture

### 1. Delta Lake Implementation
```python
from delta.tables import DeltaTable

# Write to Delta table
df.write.format("delta").save("/path/to/delta-table")

# Update Delta table
deltaTable = DeltaTable.forPath(spark, "/path/to/delta-table")
deltaTable.update(
    condition = "category = 'A'",
    set = { "value": "new_value" }
)

# Time travel
df_old = spark.read.format("delta") \
    .option("versionAsOf", "0") \
    .load("/path/to/delta-table")
```

### 2. Data Lake Organization
```
data_lake/
├── bronze/          # Raw data
│   ├── logs/
│   └── transactions/
├── silver/          # Cleaned data
│   ├── features/
│   └── labels/
└── gold/            # Analysis-ready data
    ├── training/
    └── inference/
```

## Performance Optimization

### 1. Spark Tuning
```python
# Memory configuration
spark.conf.set("spark.executor.memory", "4g")
spark.conf.set("spark.driver.memory", "2g")
spark.conf.set("spark.memory.offHeap.enabled", "true")
spark.conf.set("spark.memory.offHeap.size", "1g")

# Shuffle configuration
spark.conf.set("spark.sql.shuffle.partitions", "200")
spark.conf.set("spark.default.parallelism", "100")
```

### 2. Data Skew Handling
```python
# Salting technique for skewed joins
def add_salt(df, salt_column, num_salts):
    return df.withColumn(
        salt_column,
        (rand() * num_salts).cast("int")
    )

# Apply salting to large table
df_salted = add_salt(large_df, "salt", 10)
```

## Best Practices

### 1. Data Organization
- Implement data lakehouse architecture
- Use appropriate file formats (Parquet, ORC)
- Implement proper partitioning
- Maintain data quality
- Version control datasets

### 2. Performance
- Optimize cluster resources
- Handle data skew
- Implement caching strategies
- Monitor job performance
- Use appropriate serialization

### 3. Development
- Use version control
- Implement CI/CD
- Write unit tests
- Document code
- Monitor resource usage

## Resources

### Books
- "Hadoop: The Definitive Guide"
- "Learning Spark"
- "Designing Data-Intensive Applications"

### Online Resources
- Apache Spark Documentation
- Hadoop Documentation
- Delta Lake Documentation
- Dask Documentation

### Tools
- Spark UI
- YARN Resource Manager
- Ganglia Monitoring
- Jupyter Notebooks

## Assessment Questions

1. How do you handle data skew in Spark?
2. What are the best practices for data lake organization?
3. How do you optimize Spark performance?
4. When should you use Dask vs Spark?

## Projects

1. Build a data lake architecture
2. Implement a streaming pipeline
3. Create a distributed ML system
4. Develop a data quality framework
