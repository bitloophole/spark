from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("UDFTransformation").getOrCreate()

from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType

# Define and register UDF
def multiply_by_three(x):
    return x * 3

triple_udf = udf(multiply_by_three, IntegerType())

# Create DataFrame
numbers = [(4,), (5,), (6,)]
columns = ["value"]
df_numbers = spark.createDataFrame(numbers, columns)

# Apply UDF
df_triple = df_numbers.withColumn("tripled_value", triple_udf("value"))
print("DataFrame after applying UDF:")
df_triple.show()

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, pandas_udf
from pyspark.sql.types import IntegerType
import pandas as pd

@pandas_udf("int")
def subtract_two(s: pd.Series) -> pd.Series:
    return s - 2

df_final = df_triple.withColumn("subtracted_value", subtract_two("tripled_value"))
print("Final DataFrame after applying pandas UDF:")
df_final.show()

spark.stop()
