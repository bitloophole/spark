from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split, col, udf, concat_ws, collect_list, lit
from pyspark.sql.types import StringType

FIRST = "Khushboo"
LAST = "Kumari"

def decorate(word: str) -> str:
    # Keep the original word’s case, wrap with first+last in lowercase (as the example shows)
    return f"{FIRST}{word}{LAST}"

decorate_udf = udf(decorate, StringType())

spark = SparkSession.builder.appName("Task1-UDF-Words").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# Read raw lines from socket
lines = spark.readStream.format("socket").option("host", "host.docker.internal").option("port", 9999).load()

# Split into words and decorate each word with the UDF
words = lines.select(explode(split(col("value"), r"\s+")).alias("word"))
decorated = words.select(decorate_udf(col("word")).alias("decorated"))

# Batch all decorated words into a single output line
batched = decorated.groupBy(lit(1)).agg(concat_ws(",", collect_list(col("decorated"))).alias("output"))
query = batched.writeStream.outputMode("complete").format("console").option("truncate", False).start()

query.awaitTermination()
