from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, isnan

#Creating Spark Session
spark = (
    SparkSession.builder
    .master("local")
    .appName("Titanic data")
    .getOrCreate()
)


#Loading Dataset
dataset = spark.read.csv("/opt/spark-apps/dataset/train.csv", header=True, inferSchema=True)

dataset = dataset.select(col("Survived").cast('double'), col("Pclass").cast('double'), col("Sex"), col("Age").cast('double'), col("Fare").cast('double'), col("Embarked"))

dataset.show(5)

cols_to_check = ["Survived", "Pclass", "Sex", "Age", "Fare", "Embarked"]

# Remove any row that has NULL or NaN in these columns
for c in cols_to_check:
    dataset = dataset.filter(~(isnan(col(c)) | col(c).isNull()))

# Verify cleaned data  
dataset.select([
    count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in cols_to_check
]).show()




#imporing functions and types
import pyspark.sql.functions as F
import pyspark.sql.types as T

#importing necessary ML libraries
from pyspark.ml.feature import StringIndexer, OneHotEncoder

#importing VectorAssembler
from pyspark.ml.feature import VectorAssembler

#importing RandomForestClassifier
from pyspark.ml.classification import RandomForestClassifier

# Using Pipeline
from pyspark.ml import Pipeline

(train_df, test_df) = dataset.randomSplit([0.8, 0.2], seed=11)
print("Training Dataset Count: " + str(train_df.count()))
print("Test Dataset Count: " + str(test_df.count()))

#Lable Encoding of categorical columns
sex_indexer = StringIndexer(inputCol="Sex", outputCol="Gender")
Embarked_indexer = StringIndexer(inputCol="Embarked", outputCol="Boarded")

#Assembling all features with VectorAssembler
inputCols = ["Pclass", "Age", "Fare", "Gender", "Boarded"]
outputCol = "features"
vector_assembler = VectorAssembler(inputCols=inputCols, outputCol=outputCol)

#Modeling using Desision Tree Classifier
dt_model = RandomForestClassifier(labelCol="Survived", featuresCol="features")

#Creating Pipeline
pipeline = Pipeline(stages=[sex_indexer, Embarked_indexer, vector_assembler, dt_model])

#Fitting the model
final_pipeline = pipeline.fit(train_df)

#Making predictions
test_predictions_from_pipeline = final_pipeline.transform(test_df)
test_predictions_from_pipeline.show(5, truncate=False)

