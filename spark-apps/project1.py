from pyspark.sql import SparkSession

#Creating Spark Session
spark = SparkSession.builder \
    .appName("Titanic-MLlib-Project1") \
    .getOrCreate()

#Loading Dataset
dataset = spark.read.csv("/opt/spark-apps/dataset/train.csv", header=True, inferSchema=True)
dataset.show(5)

#Casting columns to appropriate datatypes
from pyspark.sql.functions import col

dataset = dataset.select(col("Survived").cast('float'), col("Pclass").cast('float'), col("Sex"), col("Age").cast('float'), col("Fare").cast('float'), col("Embarked"))

dataset.show(4)


#removing null values

from pyspark.sql.functions import isnull, when, count, col

dataset.select([count(when(isnull(c), c)).alias(c) for c in dataset.columns]).show()

dataset = dataset.na.drop()

dataset.select([count(when(isnull(c), c)).alias(c) for c in dataset.columns]).show()

#Converting categorical columns to numerical columns

dataset.show(3)
from pyspark.ml.feature import StringIndexer

dataset = StringIndexer(inputCol="Sex", outputCol="Gender", handleInvalid= "keep").fit(dataset).transform(dataset)
dataset = StringIndexer(inputCol="Embarked", outputCol="Boarded", handleInvalid= "keep").fit(dataset).transform(dataset)

dataset.show(2)

#Dropping original categorical columns
dataset = dataset.drop("Sex").drop("Embarked")

#Assembling features with VectorAssembler
from pyspark.ml.feature import VectorAssembler

required_featured = ["Pclass", "Age", "Fare", "Gender", "Boarded"]
assembler = VectorAssembler(inputCols=required_featured, outputCol="features")
transformed_data = assembler.transform(dataset)

transformed_data.show(5)


#splitting data
(training_data, test_data) = transformed_data.randomSplit([0.8, 0.2])
print("Training Dataset Count: " + str(training_data.count()))
print("Test Dataset Count: " + str(test_data.count()))

from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(labelCol="Survived", featuresCol="features", maxDepth=5)

model = rf.fit(training_data)
predictions_training = model.transform(training_data)
predictions_test = model.transform(test_data)

#Evaluating the model
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions_training)
print('Training Accuracy = ', accuracy)

accuracy = evaluator.evaluate(predictions_test)
print("Test Accuracy = %g " % (accuracy))

evaluator2 = MulticlassClassificationEvaluator(labelCol="Survived")
area_under_curve_training = evaluator2.evaluate(predictions_training)

print("Area Under Curve = %g " % (area_under_curve_training))

evaluator3 = MulticlassClassificationEvaluator(labelCol="Survived")
area_under_curve_test = evaluator3.evaluate(predictions_test)
print("Area Under Curve = %g " % (area_under_curve_test))



u8r3804u923
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
test_predictions_from_pipeline.select("Survived", "prediction").show(10)