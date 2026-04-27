import re
import string

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder



spark = SparkSession.builder \
    .appName("NewsCategoryClassification") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")



def clean_text_udf_fn(text):
    if text is None:
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

clean_udf = F.udf(clean_text_udf_fn, StringType())



def load_data(filepath: str):
    if filepath.endswith('.json') or filepath.endswith('.jsonl'):
        df = spark.read.json(filepath)
    else:
        df = spark.read.csv(filepath, header=True, inferSchema=True)


    if 'headline' in df.columns and 'short_description' in df.columns:
        df = df.withColumn(
            'text',
            F.concat_ws(' ', F.col('headline'), F.col('short_description'))
        )
    return df



def preprocess(df):
    
    df = df.withColumn('text_clean', clean_udf(F.col('text')))

    
    df = df.filter(F.size(F.split(F.col('text_clean'), ' ')) >= 5)

  
    df = df.dropna(subset=['text_clean', 'category'])

    
    print("\nClass distribution:")
    df.groupBy('category').count().orderBy('count', ascending=False).show(50, truncate=False)

    return df



def build_spark_pipeline():
   
    label_indexer = StringIndexer(inputCol='category', outputCol='label', handleInvalid='keep')
    tokenizer = Tokenizer(inputCol='text_clean', outputCol='words')
    remover = StopWordsRemover(inputCol='words', outputCol='filtered_words')
    hashing_tf = HashingTF(inputCol='filtered_words', outputCol='raw_features', numFeatures=20000)
    idf = IDF(inputCol='raw_features', outputCol='features', minDocFreq=2)

    
    lr = LogisticRegression(
        featuresCol='features',
        labelCol='label',
        maxIter=100,
        regParam=0.1,          
        elasticNetParam=0.0,   
        family='multinomial'
    )

    pipeline = Pipeline(stages=[label_indexer, tokenizer, remover, hashing_tf, idf, lr])
    return pipeline



def run_with_crossvalidation(pipeline, train_df):
    param_grid = ParamGridBuilder() \
        .addGrid(pipeline.getStages()[-1].regParam, [0.01, 0.1, 0.5]) \
        .build()

    evaluator = MulticlassClassificationEvaluator(
        labelCol='label',
        predictionCol='prediction',
        metricName='accuracy'
    )

    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=5,
        seed=42
    )

    print("\nRunning 5-fold cross-validation (this may take a while)...")
    cv_model = cv.fit(train_df)
    best_model = cv_model.bestModel

    print(f"Best regParam: {best_model.stages[-1].getRegParam()}")
    return best_model

if __name__ == '__main__':
    import sys

    filepath = sys.argv[1] if len(sys.argv) > 1 else '../dataset/news.json'
    output_path = sys.argv[2] if len(sys.argv) > 2 else './output'

    print(f"Loading data from: {filepath}")
    df = load_data(filepath)
    df = preprocess(df)
    df.cache() 

   
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    print(f"\nTrain size: {train_df.count():,} | Test size: {test_df.count():,}")

    pipeline = build_spark_pipeline()
    best_model = run_with_crossvalidation(pipeline, train_df)

   
    test_preds = best_model.transform(test_df)
    evaluator = MulticlassClassificationEvaluator(
        labelCol='label', predictionCol='prediction', metricName='accuracy'
    )
    test_acc = evaluator.evaluate(test_preds)
    print(f"\nTest Accuracy: {test_acc:.4f}")

   
    df.write.mode('overwrite').parquet(f'{output_path}/processed_news.parquet')
    print(f"\nProcessed data saved to '{output_path}/processed_news.parquet'")

    spark.stop()
