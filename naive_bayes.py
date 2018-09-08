from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

iris_df = spark.read.csv("../data/iris.data.csv", inferSchema=True)

iris_df.take(1)
[Row(_c0=5.1, _c1=3.5, _c2=1.4, _c3=0.2, _c4='Iris-setosa')]

iris_df = iris_df.select(col(“_c0”).alias(“sepal_length”),
	col(“_c1”).alias(“sepal_width”),
	col(“_c2”).alias(“petal_length”),
	col(“_c3”).alias(“petal_width”),
	col(“_c4”).alias(“species”)
)

# Change the number of each column to more meaningful names
# merges the columns into a single feature column called “features”
vectorAssembler = VectorAssembler(inputCols=[“sepal_length”, “sepal_width”, “petal_length”, “petal_width”], outputCol = “features”)

# transforms the features into the feature column as specified in the previous code
viris_df = vectorAssembler.transform(iris_df)

viris_df.take(1)

# create an indexer using the transformation StringIndexer
# input column is species, because that's the string 
# And, our output column, will be called label
indexer = StringIndexer(inputCol = “species”, outputCol= “label”)

# create a data frame that captures this indexed value
iviris_df = indexer.fit(viris_df).transform(viris_df)

# view the new indexed and vectorized data frame created
iviris_df.take(1)

# split the data frame into one set with 60% of the data, 
# and the other with 40% of the data. “1” for the seed
splits = iviris_df.randomSplit([0.6, 0.4],1)

train_df = splits[0]

test_df = splits[1]

# create a Naïve Bayes model, and instead of binary model, 
# since we have more than 2 labels, 
# we will choose multinomial labels
nb = NaiveBayes(modelType = “multinomial”)

# fit the data to the model
nbmodel = nb.fit(train_df)

# once we built the model and fit it with our training data, 
# we can use the model to make predictions. 
# To do so, we can transform the test data on the nbmodel we created.
predictions_df = nbmodel.transform(test_df)

# take a look at the dataframe now, 
# there will be some columns added with a final column called “label”
# in this case the label is 0.0 which means the model 
# predicts that the test example passed belongs to the first iris species
predictions_df.take(1)

# Evaluator for Multiclass Classification, which 
# expects two input columns: prediction and label.
# MetricName is accuracy because we trying to measure 
# the accuracy between the actual labels and predictions.
evaluator = MulticlassClassificationEvaluator(labelCol= “label” , predictionCol = “prediction” , metricName= “accuracy”)

# will take in the prediction column from the predictions 
# dataframe and evaluate the accuracy versus the label column 
nbaccuracy = evaluator.evaluate(predictions_df)

# will show the accuracy, which is in this case, is ~58%
print(nbaccuracy)