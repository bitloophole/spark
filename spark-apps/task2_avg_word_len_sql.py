from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split, col, length

spark = SparkSession.builder.appName("Task2-Avg-Word-Length-SQL").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

lines = spark.readStream.format("socket").option("host", "host.docker.internal").option("port", 9999).load()
words = lines.select(explode(split(col("value"), r"\s+")).alias("word")).where(col("word") != "")

# Create/replace a temp view on the streaming DataFrame
words.createOrReplaceTempView("words_stream")

# Use SQL to compute average length
avg_len = spark.sql("""
  SELECT AVG(LENGTH(word)) AS avg_word_length
  FROM words_stream
""")

# complete mode shows the current aggregate each trigger
query = avg_len.writeStream.outputMode("complete").format("console").option("truncate", False).start()
query.awaitTermination()
