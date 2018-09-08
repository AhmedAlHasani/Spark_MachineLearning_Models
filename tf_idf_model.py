from pyspark.ml feature import HashingTF, IDF

# Sentence dataframe
sentences_df = spark.createDataFrame(
	[
		(1, "This is an introduction to Spark MLlib"),
		(2, "MLlib includes libraries for classification and regression"),
		(3, "It also contains supporting tools for pipelines")
	],
		[“id”, “sentence”]
	)

sent_token = Tokenizer(inputCol = 'sentence', output = 'words')

sent_tokenized_df = sent_token.transform(sentences_df)

# Maps a sequence of terms to their term frequencies using the hashing trick. 
# We need to tell it how many features we want to keep track of.
hashingTF = HashingTF(inputCol = 'words', outputCol= 'rawFeatures', numFeatures = 20)

# transform the data, take in the tokenized sentences 
# and give us a hashmap (dictionary) with frequencies
sent_hfTF_df = hashingTF.transform(sent_tokenized_df)

sent_hfTF_df.take(1)

# Return a map of each word, represented by an index (e.g. this = 1), and the frequency? 
[Row(
	id=1, sentence='This is an introduction to Spark MLlib', 
	words=['this', 'is', 'an', 'introduction', 'to', 'spark', 'mllib'], 
	rawFeatures=SparseVector(20, {1: 2.0, 5: 1.0, 6: 1.0, 8: 1.0, 12: 1.0, 13: 1.0})
)]

# Compute the Inverse Document Frequency (IDF) given a collection of documents.
idf = IDF(inputCol = 'rawFeatures', outputCol= 'idf_features')

# This will scale the rawFeature vector values based on how often the 
# words appear in the entire collection of sentences.
idfModel = idf.fit(sent_hfTF_df)

# to create a new dataframe with IDF features column. 
# So to do that I'm going to specify tfidf_df.  
# this is our dataframe that has both the term frequency 
# and the inverse document frequency transformations applied, 
# and I'm going to create that by using the idfModel.
tfidf_df = idfModel.transform(sent_hfTF_df)