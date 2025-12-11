from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("RetailStoreInsights").getOrCreate()
data = [
   ('Ulysses','Book',23.17,16), ('Apple','Fruit',2.34,8),
   ('Pineapple','Fruit',2.57,1), ('Apple','Fruit',2.43,6),
   ('To Kill a Mockingbird','Book',24.14,19),
   ('To Kill a Mockingbird','Book',11.18,11),
   ('Watermelon','Fruit',3.35,15), ('Pride and Prejudice','Book',24.99,3),
   ('To Kill a Mockingbird','Book',21.82,17), ('Moby Dick','Book',14.83,20),
   ('Pride and Prejudice','Book',5.03,16), ('Jane Eyre','Book',20.40,8),
   ('Moby Dick','Book',5.55,20), ('Don Quixote','Book',19.75,17),
   ('Watermelon','Fruit',2.31,9), ('Hamlet','Book',18.20,12),
   ('Mango','Fruit',4.10,7), ('1984','Book',16.75,14),
   ('Strawberry','Fruit',1.90,25), ('War and Peace','Book',22.50,9),
   ('Orange','Fruit',3.05,13), ('The Great Gatsby','Book',12.30,10),
   ('Peach','Fruit',2.80,11), ('Grapes','Fruit',2.60,18),
   ('Pride and Prejudice','Book',9.50,5)
]

columns = ["product_name","category","price","quantity"]
df = spark.createDataFrame(data, columns)

df.show(10, truncate=False)
df.printSchema()

# Select only product name and price
df.select("product_name","price").show()

# Products priced above $2
df.filter(df.price > 2).show()

# Count per category
df.groupBy("category").count().show()

# Average price of all products
from pyspark.sql.functions import avg
df.select(avg("price").alias("avg_price")).show()

# Add a discounted price column (10% off)
from pyspark.sql.functions import round, col
df = df.withColumn("discounted_price", round(col("price")*0.9,2))
df.show(10)



df.createOrReplaceTempView("retail_sales")

# Total number of products sold (counting duplicates)
spark.sql("SELECT COUNT(*) AS total_products FROM retail_sales").show()

# Total sales (sum of prices) for each category
spark.sql("""
    SELECT category, ROUND(SUM(price),2) AS total_sales
    FROM retail_sales
    GROUP BY category
""").show()

# (Optional) Frequency of each product
spark.sql("""
    SELECT product_name, COUNT(*) AS frequency
    FROM retail_sales
    GROUP BY product_name
    ORDER BY frequency DESC
""").show()
