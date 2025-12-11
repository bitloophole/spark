from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split, trim, col

spark = SparkSession.builder.appName("Task3-Even-Numbers-DF").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

lines = (
    spark.readStream
         .format("socket")
         .option("host", "host.docker.internal")
         .option("port", 9999)
         .load()
)

tokens = lines.select(explode(split(col("value"), r"\s+")).alias("raw"))
tokens = tokens.select(trim(col("raw")).alias("raw")).where(col("raw") != "")

# keep only numeric tokens, cast to int/bigint, then filter evens
nums = tokens.where(col("raw").rlike(r'^[+-]?\d+$')) \
             .select(col("raw").cast("bigint").alias("n")) \
             .where((col("n") % 2) == 0)

query = (
    nums.writeStream
        .outputMode("append")
        .format("console")
        .option("truncate", False)
        .start()
)

query.awaitTermination()
