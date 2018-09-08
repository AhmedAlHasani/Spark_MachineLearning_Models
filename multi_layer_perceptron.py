from pyspark.ml.classification import MultilayerPerceptronClassifier

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

# the first layer is 4, because we have 4 inputs. The last layer is 3
# because we have three types of labels (3 types of iris to classify test data). 
# We have two “hidden” layers or middle layers, each with 5 neurons.
layers = [4,5,5,3]

# mlp will be our instance of the multi-layer perceptron classifier
# the layers mlp will have are determined by the layers list we created
# the seed is set to 1, since multi-layer perceptron uses a random number generator
mlp = MultilayerPerceptronClassifier(layers = layers, seed=1)

# we will have an 'mlp' model that will fit the training data to itself
mlp_model = mlp.fit(train_df)

# to make predictions, we will use the model we built with the training data 
# and transform it with the test data to make predictions.
mlp_predictions = mlp_model.transform(test_df)
 
# create an evaluator to evaluate multiclasses
# the metric name is accuracy
# since we want to measure the accuracy of the predictions. 
mlp_evaluator = MulticlassClassificationEvaluator(metricName= “accuracy”)

# mlp accuracy will hold the accuracy for our model. 
# We call the mlp_evaluator object to evaluate our mlp_predictions.
mlp_accuracy = mlp_evaluator.evaluate(mlp_predictions)
 
# ~93%
mlp_accuracy