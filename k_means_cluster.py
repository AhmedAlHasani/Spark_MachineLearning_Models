from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans

# create a df from the csv file
# with the header option true and also infer the schema of the CSV file.
cluster_df = spark.read.csv("../data/clustering_dataset.csv" , header=True, interSchema=True) 

# verify that it has 3 columns as is in csv file
cluster_df

# will show the 3 columns. Each 25 rows are clustered into a range of numbers
cluster_df.show(75)

# A feature transformer that merges multiple columns into a vector column.
vectorAssembler = VectorAssembler(inputCols = [“col1”, “col2”, “col3”], outputCol = “features”)

# vcluster = vectorized cluster data frame
# the transformer will give us a new feature column = “features” which is needed because
# the k means algorithm will work with the new column
vcluster_df = vectorAssembler.transform(cluster_df)

# create an object kmeans , which will have a cluster of 3
kmeans = KMeans().setK(3)

# this will determine where the kmeans algorithm starts
# will give us consistency during testing
kmeans = kmeans.setSeed(1)

# create a model called kmodel which will be built using KMeans and 
# the data to fit to this model is vcluster_df
# because it contains the feature vector
kmodel = kmeans.fit(vcluster_df)

# explore the centers of the model, for each cluster. It will give 3 centers for each cluster. 
# It will give for each 25 rows, for each 3 clusters in this exercise.
centers = kmodel.clusterCenters()