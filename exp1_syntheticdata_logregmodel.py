import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn import metrics

import seaborn as sns
import pandas as pd
import random
import csv

# df_clean = pd.read_csv('fix_after_output.csv')

def train_model(X_train, X_test, y_train,y_test):
  # instantiate the model (using the default parameters)
  model1 = LogisticRegression(max_iter=1000)
  # fit the model with data
  model1.fit(X_train,y_train.values.ravel())#y_train.values.ravel() because a 1d array is expected, not a column
  print(y_train)
  return model1

def get_train_acc(model, X_train, y_train):
  train_acc = model1.score(X_train, y_train)
  print("The Accuracy for Training Set is {}".format(train_acc*100))
  return train_acc

def get_test_acc(model, X_test, y_test):
  test_acc = model1.score(X_test, y_test)
  print("The Accuracy for Testing Set is {}".format(test_acc*100))
  return test_acc

def cnf_matrix(model,X_test,y_test):
  y_pred=model.predict(X_test)
  cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
  print(cnf_matrix)
  return cnf_matrix



  
  



#**************************************************************************************************************************
#TRIED TO USE CROSS VALIDATION, BUT PROBABLY DID SOMETHING WRONG ALONG THE WAY. GOT LOWER ACCURACY USING CV THAN WITHOUT?
# logreg = LogisticRegression(max_iter = 5000)
# predicted = cross_val_predict(logreg, X, y.values.ravel(), cv=2)
# print(metrics.accuracy_score(y, predicted))
#**************************************************************************************************************************



