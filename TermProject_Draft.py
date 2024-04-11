#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd

# Assumption : We have a fatalities column with value of 0 or 1
# Need to determine which columns are features
# Will need to get columns in source data reduced to only what is being used, or 
# Write a way to only select the relevant ones.

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("MLP NYC") \
    .getOrCreate()

df = pd.read.csv("Put Path Here")

train_data, test_data = df.randomSplit([0.7, 0.3], seed=123)

# Define input features
feature_columns = [col for col in train_data.columns if col != 'fatality']

# If we want to include all the features into one column, the vector assembler
# Can do this, otherwise comment it out!
# Vector Assembler
vector_assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
train_data = vector_assembler.transform(train_data)
test_data = vector_assembler.transform(test_data)

#This is a fairly standard...standardization rpocess
# Standardize features
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=False)
scaler_model = scaler.fit(train_data)
train_data = scaler_model.transform(train_data)
test_data = scaler_model.transform(test_data)

# Define Multi-Layer Perceptron classifier
# If we aren't using vectorizing, we need to manually change the first layer to the number of features
# Input layer, two hidden layers, output layer
layers = [len(feature_columns), 20, 20, 20, 1]  
mlp_classifier = MultilayerPerceptronClassifier(layers=layers, seed=42)

# Train the model
mlp_model = mlp_classifier.fit(train_data)

# Make predictions
predictions = mlp_model.transform(test_data)

# Evaluate the model
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Accuracy = {:.2f}%".format(accuracy * 100))

# Stop SparkSession
spark.stop()


# In[ ]:


from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Create a SparkSession
spark = SparkSession.builder \
    .appName("LogReg Classifier NYC") \
    .getOrCreate()
#Assumption: Need to determine what the label vs the features are
df = spark.createDataFrame(data, ["label", "features"])

#Again, adding all the features into a single column for easier use.
# Assemble features into a single column
assembler = VectorAssembler(inputCols=["features"], outputCol="features_vector")
df = assembler.transform(df)

# Split data into training and test sets
train_data, test_data = df.randomSplit([0.7, 0.3], seed=123)

# Create Logistic Regression model
lr = LogisticRegression(featuresCol="features_vector", labelCol="label")

# Train the model
lr_model = lr.fit(train_data)

# Make predictions on the test data
predictions = lr_model.transform(test_data)

