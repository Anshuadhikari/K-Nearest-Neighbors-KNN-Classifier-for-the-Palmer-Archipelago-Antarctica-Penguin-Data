#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 11:26:21 2023

@author: pchandra
"""
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
import pandas as pd 



def train(X, y, train_index, k):
    # X = of type dataframe that holds all the features
    # y = of type Series that holds labels corresponding to the rows in the X df
    # train_index = array of indices of rows selected to be part of training
    # k = number of neighbors to consider   

    # Split the data
    train_X = X.iloc[train_index]
    train_Y = y.iloc[train_index]

    # Normalize train_X
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)

    # Train model
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(train_X, train_Y)

    return neigh

#Function to predict()
def predict(X, y, test_index, neigh):
     
 #Predictions 
 predictions = []
 
 #Split the data
 test_X = X.iloc[test_index]
 print("Number of test: ", len(test_X))
 test_Y = y.iloc[test_index]
 
 #Normalize test_X
 scaler = StandardScaler()
 test_X = scaler.fit_transform(test_X)
 
 #Predictions
 predictions = neigh.predict(test_X)
 
 return test_Y, predictions


'''
Function calculates F1-score and returns score 
''' 
def evaluation(true_values, predicted_values):
    score = f1_score(true_values, predicted_values, average='weighted')
    return score

'''
KNN Classifier: Classifier that predicts and returns average f1_score
'''
'''
 KNN Classifier: Classifier that predicts and returns average f1_score
'''
def knn_class(k, df, features, target_column_name):
    X = df[features]
    y = df[target_column_name]

    # Initialize array to store f1 scores
    f1_scores = []

    # Setup KFold cross validation
    num_folds = 10
    kf = KFold(n_splits=num_folds, shuffle=True)

    for i, (train_index, test_index) in enumerate(kf.split(X)):
        print("Fold: ", i+1)

        # Calling train()  
        neigh = train(X, y,train_index, k)

        # Calling test/predict()
        true_values, predicted_values = predict(X, y, test_index, neigh)

        # Calling evaluation function
        f1 = evaluation(true_values, predicted_values)

        # Append f1 score to array
        f1_scores.append(f1)

    # Calculate mean f1 score
    mean_f1_score = sum(f1_scores) / num_folds

    # Return mean f1 score
    return mean_f1_score

import pandas as pd

k_values = [1, 3, 5, 7, 9, 11]
accuracy_values = [0.5, 0.4666666666666666, 0.4666666666666666, 0.5333333333333333, 0.6, 0.7333333333333333]

#pengu = pd.DataFrame({'K Value': k_values, 'Accuracy': accuracy_values})
#print(pengu)


# Load dataset
df = pd.read_csv("penguins_size.csv")
print(df.columns)
df = df.dropna()
# Features
features = ['culmen_depth_mm', 'flipper_length_mm']

# Target column name
target_column_name = 'island'

best_k = None
best_f1_score = -1

# Loop over odd values of k from 1 to 40
for k in range(1, 41, 2):
    # Do KNN-based classification
    prediction_score = knn_class(k, df, features, target_column_name)
    
    # Update best k and f1 score if current score is higher
    if prediction_score is not None and prediction_score > best_f1_score:
        best_k = k
        best_f1_score = prediction_score
    
    print("KNN for k =", k, "is:", prediction_score)

# Print the best k and f1 score
print("Best k is:", best_k, "with f1 score of:", best_f1_score)
