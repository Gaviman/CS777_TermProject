#!/usr/bin/env python
# coding: utf-8

# In[2]:


from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
spark = SparkSession.builder.appName("Motor").getOrCreate()
import tkinter as tk
from tkinter import messagebox
from pyspark.sql.types import *
from pyspark.sql.functions import col, when
import time 


# In[4]:


def process_data():
# Check arguments
    if len(sys.argv) != 4:
        print("Usage: term_project_preprocessing.py <input_file_path> <output_dir> ", file=sys.stderr)
        exit(-1)
    
#read in our motor accident file
    motor_df = spark.read.option("header",True).csv(sys.argv[1])

#import col to manipulate columns more easily
    from pyspark.sql.functions import col

#read in our zip code file and filter for just zip code, median income, and median property value
    property_df = property_df = spark.read.option("header",True).csv(sys.argv[2])

#import when in order to create columns
    from pyspark.sql.functions import when

#create columns 'injury' and 'fatality' - binary categorical variables
#for whether, respectively, there were any injuries or fatalities

    motor_df = motor_df.withColumn("injury", \
                               when(col("NUMBER OF PERSONS INJURED")>0, 1).otherwise(0))

#now create same type of column for fatality
    motor_df = motor_df.withColumn("fatality", \
                                   when(col("NUMBER OF PERSONS KILLED")>0, 1).otherwise(0))
    

#rename the column "ZIP CODE" in motor_df to "acc_zip_code" before joining
    motor_df = motor_df.withColumnRenamed("ZIP CODE", "acc_zip_code")

#do an inner join on the zip code column
    motor_with_median_home_value_df = \
        motor_df.join(property_df,\
                      motor_df.acc_zip_code == property_df.zip_code,"inner")

#save to csv file
    motor_with_median_home_value_df.coalesce(1).write.option("header",True).csv(sys.argv[3])


# In[7]:


def mlp():
    starttime= time.time()
# Create a DataFrame
    df = spark.read.option("header", True).csv(sys.argv[0])

    df = df.withColumn('median_home_value', col('median_home_value').cast("int"))
    df = df.withColumn('fatality', col('fatality').cast("int"))

# Combine integer features into a single feature vector
    assembler = VectorAssembler(inputCols=['median_home_value'],
                            outputCol="features")

    df = assembler.transform(df)

# Split the data into train and test sets
    train, test = df.randomSplit([0.8, 0.2], seed=1234)

# Define the MLP model
    layers = [1, 4, 3,1]  # input layer, two hidden layers, and output layer
    mlp = MultilayerPerceptronClassifier(layers=layers, seed=1234, labelCol = "fatality")

# Train the model
    model = mlp.fit(train)

# Make predictions
    predictions = model.transform(test)

# Evaluate the model
    evaluator = MulticlassClassificationEvaluator(metricName="accuracy", labelCol = 'fatality')
    accuracy = evaluator.evaluate(predictions)
    print("Test Accuracy = {:.2f}%".format(accuracy * 100))
    
    endtime = time.time()

    time_elapsed = endtime - starttime

# Reset evaluator to get different metric

    evaluator.setMetricName("f1")
    f1_score = evaluator.evaluate(predictions)

    evaluator.setMetricName("weightedPrecision")
    weighted_precision = evaluator.evaluate(predictions)

    evaluator.setMetricName("weightedRecall")
    weighted_recall = evaluator.evaluate(predictions)

    evaluator.setMetricName("weightedFMeasure")
    weighted_f1_score = evaluator.evaluate(predictions)

# Prepare results as a dictionary
    results = {
        "Accuracy": accuracy,
        "F1 Score": f1_score,
        "Weighted Precision": weighted_precision,
        "Weighted Recall": weighted_recall,
        "Weighted F1 Score": weighted_f1_score,
        "Time Elapsed": time_elapsed
    }

# Print or return the results
    for metric, value in results.items():
        print(f"{metric}: {value}")

    results_string = str(results)    
    
    output_path = sys.argv[0]

#results.saveAsTextFile(output_path, mode = 'overwrite')
    with open(output_path, "w") as text_file:
        text_file.write(results_string)


# In[9]:


def log_reg():
    starttime= time.time()
    df = spark.read.option("header", True).csv(sys.argv[0])

# Define input features

    selected_columns = ['median_home_value','fatality']
    selected_df = df.select(selected_columns)

    selected_df = selected_df.withColumn('median_home_value', col('median_home_value').cast("int"))
    selected_df = selected_df.withColumn('fatality', col('fatality').cast("int"))

    #print(selected_df)

# Assemble features into a single column
    assembler = VectorAssembler(inputCols=['median_home_value'], outputCol="features_vector")
    assembled_df = assembler.transform(selected_df)
# Split data into training and test sets
    train_data, test_data = assembled_df.randomSplit([0.7, 0.3], seed=123)

# Create Logistic Regression model
    lr = LogisticRegression(featuresCol="features_vector", labelCol="fatality")

    print(train_data)

# Train the model
    lr_model = lr.fit(train_data)

# Make predictions on the test data
    predictions = lr_model.transform(test_data)
    
# Evaluate the model
    evaluator = MulticlassClassificationEvaluator(metricName="accuracy", labelCol = 'fatality')

# Calculate various metrics
    accuracy = evaluator.evaluate(predictions)
    
    endtime = time.time()

    time_elapsed = endtime - starttime

# Reset evaluator to get different metric

    evaluator.setMetricName("f1")
    f1_score = evaluator.evaluate(predictions)

    evaluator.setMetricName("weightedPrecision")
    weighted_precision = evaluator.evaluate(predictions)

    evaluator.setMetricName("weightedRecall")
    weighted_recall = evaluator.evaluate(predictions)

    evaluator.setMetricName("weightedFMeasure")
    weighted_f1_score = evaluator.evaluate(predictions)

# Prepare results as a dictionary
    results = {
        "Accuracy": accuracy,
        "F1 Score": f1_score,
        "Weighted Precision": weighted_precision,
        "Weighted Recall": weighted_recall,
        "Weighted F1 Score": weighted_f1_score,
        "Time Elapsed": time_elapsed
    }

# Print or return the results
    for metric, value in results.items():
        print(f"{metric}: {value}")

    results_string = str(results)    
    
    output_path = sys.argv[1]

#results.saveAsTextFile(output_path, mode = 'overwrite')
    with open(output_path, "w") as text_file:
        text_file.write(results_string)


# In[ ]:


def onSubmit(zip_code):
    #find a matching zip code
    target_row = df.filter((df.col("col1") == zip_code)).head()
    #pull the median home value
    median_value = target_row.col('median_home_value')
    #Calculate Log Reg
    log_reg_prediction = lr_model.transform(median_value)
    mlp_prediction = model.transform(median_value)
    
    #return text
    result_label.config(text="Logistic Regression Prediction: " + log_reg_prediction +
                       "Multi Layer Preceptron Prediction: " + mlp_prediction)


# In[10]:


# Create the main window
root = tk.Tk()
root.title("NYC Crash Fatality Prediction")

# Create label and entry widgets
label_address = tk.Label(root, text="Enter ZipCode:")
label_address.pack()

entry_address = tk.Entry(root, width=50)
entry_address.pack()

# Create Process button
process_button = tk.Button(root, text="Process Data", command=process_data)
process_button.pack()

# Create MLP button
mlp_button = tk.Button(root, text="MLP Regression", command=mlp)
mlp_button.pack()

# Create Process button
log_reg_button = tk.Button(root, text="Logistic Regression", command=log_reg)
log_reg_button.pack()

# Create submit button
submit_button = tk.Button(root, text="Probability for Given Zipcode", command=on_submit)
submit_button.pack()

# Create a label to display the result
result_label = tk.Label(window, text="")
result_label.pack()

# Run the main event loop
root.mainloop()


# In[ ]:


spark.stop()


# In[ ]:




