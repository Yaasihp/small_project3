from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import time

######################### 
######################### 
##*****************************##
#1- Initializing Spark session
######################### 
spark = SparkSession.builder \
    .appName("Diabetes_Classification_SparkML") \
    .master("spark://hadoop1:7077") \
    .config("spark.executor.memory", "2g") \
    .config("spark.executor.cores", "2") \
    .getOrCreate()

######################### 
######################### 
##*****************************##
#2- loading the data
######################### 
path = "/home/sat3812/spark_project3/diabetes_binary_health_indicators_BRFSS2015.csv"
df = spark.read.csv(path, header=True, inferSchema=True)

print(f"Total records: {df.count()}")
print("Schema:")
df.printSchema()

# target column
label_col = "Diabetes_binary"

# checking label is DoubleType
df = df.withColumn(label_col, F.col(label_col).cast(DoubleType()))


######################### 
######################### 
##*****************************##
#3- Feature partition: binary vs continuous
######################### 
# numeric columns(not target)
numeric_cols = [c for (c, t) in df.dtypes if c != label_col and t in ("int", "bigint", "double", "float", "smallint")]

# per-column stats to separate binary (0/1, ~2 distinct) vs continuous
agg_exprs = []
for c in numeric_cols:
    agg_exprs.extend([
        F.min(F.col(c)).alias(f"{c}__min"),
        F.max(F.col(c)).alias(f"{c}__max"),
        F.approx_count_distinct(F.col(c)).alias(f"{c}__adist")
    ])

stats_row = df.agg(*agg_exprs).collect()[0].asDict()

binary_cols, continuous_cols = [], []
for c in numeric_cols:
    cmin = stats_row.get(f"{c}__min")
    cmax = stats_row.get(f"{c}__max")
    cadist = stats_row.get(f"{c}__adist")
    # Heuristic: treat as binary if values are in {0,1} (min==0,max==1) and only ~2 distinct values
    if cmin == 0 and cmax == 1 and cadist is not None and cadist <= 2:
        binary_cols.append(c)
    else:
        continuous_cols.append(c)

print("\nDetected feature groups:")
print(f"  Binary (unscaled): {len(binary_cols)} cols")
print(f"  Continuous (scaled): {len(continuous_cols)} cols")

# --------------------------------
# 4) Class weights (imbalance fix)
# --------------------------------
# checking class balance:
# w(target) = N_total / (2 * N_label)
counts_rows = df.groupBy(label_col).count().collect()
counts = {float(r[label_col]): int(r['count']) for r in counts_rows}
total_n = sum(counts.values())
w_map = {lbl: (total_n / (2.0 * cnt)) for lbl, cnt in counts.items()}

weight_expr = F.create_map(
    *[x for kv in w_map.items() for x in (F.lit(float(kv[0])), F.lit(float(kv[1])))]
).getItem(F.col(label_col))


df_w = df.withColumn("classWeight", weight_expr)



######################### 
######################### 
##*****************************##
#5- startifies
#########################


# sampleBy to take 80% from each class as train, anything remaining is test.
fractions = {float(lbl): 0.8 for lbl in counts.keys()}
train_df = df_w.sampleBy(label_col, fractions=fractions, seed=42)
# Subtract to get the complementary test set 
test_df = df_w.subtract(train_df)

print(f"\nStratified split:")
print(f"  Train count: {train_df.count()}")
print(f"  Test  count: {test_df.count()}")
print("  Train label distribution:")
train_df.groupBy(label_col).count().show()
print("  Test label distribution:")
test_df.groupBy(label_col).count().show()

######################### 
######################### 
##*****************************##
#6. Pipeline +scaling 
#########################

stages = []

assembler_bin = None
assembler_cont = None
inputs_for_final = []

if continuous_cols:
    assembler_cont = VectorAssembler(inputCols=continuous_cols, outputCol="cont_features", handleInvalid="keep")
    scaler = StandardScaler(inputCol="cont_features", outputCol="cont_scaled", withStd=True, withMean=False)
    stages += [assembler_cont, scaler]
    inputs_for_final.append("cont_scaled")

if binary_cols:
    assembler_bin = VectorAssembler(inputCols=binary_cols, outputCol="bin_features", handleInvalid="keep")
    stages += [assembler_bin]
    inputs_for_final.append("bin_features")

# Final features (concat scaled continuous + raw binary)
final_assembler = VectorAssembler(inputCols=inputs_for_final, outputCol="features", handleInvalid="keep")
stages.append(final_assembler)

# Logistic Regression with class weights
lr = LogisticRegression(
    featuresCol="features",
    labelCol=label_col,
    weightCol="classWeight",
    maxIter=100,
    regParam=0.0,
    elasticNetParam=0.0,
    standardization=False   
)


stages.append(lr)

pipeline = Pipeline(stages=stages)

######################### 
######################### 
##*****************************##
#7-  Train + Predict (timed)
#########################


t0 = time.time()
lr_model = pipeline.fit(train_df)
pred = lr_model.transform(test_df)
elapsed = time.time() - t0

######################### 
######################### 
##*****************************##
#8-  Metrics + Confusion Matrix
#########################

acc_eval = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName="accuracy")
roc_eval = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol="rawPrediction", metricName="areaUnderROC")
pr_eval  = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol="rawPrediction", metricName="areaUnderPR")

accuracy = acc_eval.evaluate(pred)
roc_auc  = roc_eval.evaluate(pred)
pr_auc   = pr_eval.evaluate(pred)

print(f"\n=== Logistic Regression (weighted, scaled-continuous) ===")
print(f"Train+Predict time: {elapsed:.2f} sec")
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC:  {roc_auc:.4f}")
print(f"PR  AUC:  {pr_auc:.4f}")

print("\nConfusion Matrix (label vs prediction):")
cm = (pred
      .groupBy(F.col(label_col).alias("label"), F.col("prediction").alias("pred"))
      .count()
      .orderBy("label", "pred"))
cm.show()

# Precision/Recall/F1
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
prec_eval = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName="weightedPrecision")
rec_eval  = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName="weightedRecall")
f1_eval   = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName="f1")

print(f"Weighted Precision: {prec_eval.evaluate(pred):.4f}")
print(f"Weighted Recall:    {rec_eval.evaluate(pred):.4f}")
print(f"F1 Score:           {f1_eval.evaluate(pred):.4f}")

spark.stop()

