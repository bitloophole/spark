from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, isnan

# =========================
# 1. Spark session
# =========================
spark = (
    SparkSession.builder
    .master("local")
    .appName("Titanic data")
    .getOrCreate()
)

# =========================
# 2. Load and CLEAN dataset
# =========================
dataset = spark.read.csv(
    "/opt/spark-apps/dataset/train.csv",
    header=True,
    inferSchema=True
)

# Select only needed columns and cast numerics
dataset = dataset.select(
    col("Survived").cast("double").alias("Survived"),
    col("Pclass").cast("double").alias("Pclass"),
    col("Sex"),
    col("Age").cast("double").alias("Age"),
    col("Fare").cast("double").alias("Fare"),
    col("Embarked")
)

# Columns we care about
cols_to_check = ["Survived", "Pclass", "Sex", "Age", "Fare", "Embarked"]


# Drop any row that has NULL or NaN in these columns
for c in cols_to_check:
    dataset = dataset.filter(~(isnan(col(c)) | col(c).isNull()))

# (Optional) verify cleaned
dataset.select([
    count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in cols_to_check
]).show()

dataset.printSchema()
dataset.show(5)

# =========================
# 3. Train / test split
# =========================
(train_df, test_df) = dataset.randomSplit([0.8, 0.2], seed=11)
print("Training Dataset Count: " + str(train_df.count()))
print("Test Dataset Count: " + str(test_df.count()))

# =========================
# 4. Pipeline definition
# =========================
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline

# Label encoding of categorical columns (ROBUST)
sex_indexer = StringIndexer(
    inputCol="Sex",
    outputCol="Gender",
    handleInvalid="keep"      # <-- prevent crashes from unseen/invalid categories
)

Embarked_indexer = StringIndexer(
    inputCol="Embarked",
    outputCol="Boarded",
    handleInvalid="keep"
)

# Assembling all features with VectorAssembler (ROBUST)
inputCols = ["Pclass", "Age", "Fare", "Gender", "Boarded"]
vector_assembler = VectorAssembler(
    inputCols=inputCols,
    outputCol="features",
    handleInvalid="skip"      # <-- drop bad rows instead of throwing FAILED_EXECUTE_UDF
)

# Modeling using RandomForestClassifier
dt_model = RandomForestClassifier(
    labelCol="Survived",
    featuresCol="features",
    maxDepth=5
)

# Creating Pipeline
pipeline = Pipeline(stages=[sex_indexer, Embarked_indexer, vector_assembler, dt_model])

# =========================
# 5. Fit the model
# =========================
final_pipeline = pipeline.fit(train_df)

# =========================
# 6. Making predictions
# =========================
test_predictions_from_pipeline = final_pipeline.transform(test_df)

test_predictions_from_pipeline.select(
    "Survived", "prediction", "probability", "Pclass", "Sex", "Age", "Fare", "Embarked"
).show(10, truncate=False)

# =========================
# 7. Evaluation (optional)
# =========================
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

acc_eval = MulticlassClassificationEvaluator(
    labelCol="Survived",
    predictionCol="prediction",
    metricName="accuracy"
)
test_acc = acc_eval.evaluate(test_predictions_from_pipeline)
print("Test Accuracy from pipeline = %g" % test_acc)

roc_eval = BinaryClassificationEvaluator(
    labelCol="Survived",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)
pr_eval = BinaryClassificationEvaluator(
    labelCol="Survived",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderPR"
)

print("Test Area Under ROC = %g" % roc_eval.evaluate(test_predictions_from_pipeline))
print("Test Area Under PR  = %g" % pr_eval.evaluate(test_predictions_from_pipeline))
