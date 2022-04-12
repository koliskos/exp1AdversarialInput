import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn import metrics

import seaborn as sns
import pandas as pd
import random
import csv

df_clean = pd.read_csv('fix_after_output.csv')

X_numpy = df_clean.drop("App_Number", axis=1)
X = X_numpy.drop("Approval_Status", axis=1)
y = df_clean[['Approval_Status']]


#**************************************************************************************************************************
#TRIED TO USE CROSS VALIDATION, BUT PROBABLY DID SOMETHING WRONG ALONG THE WAY. GOT LOWER ACCURACY USING CV THAN WITHOUT?
# logreg = LogisticRegression(max_iter = 5000)
# predicted = cross_val_predict(logreg, X, y.values.ravel(), cv=2)
# print(metrics.accuracy_score(y, predicted))
#**************************************************************************************************************************



# instantiate the model (using the default parameters)
model1 = LogisticRegression(max_iter=1000)
X_train, X_test, y_train, y_test = train_test_split(X, y) #nicely splits up data into training and testing data. We could stratify using y so that the random split of training/test data has same proportions of approval or rejection for Approval_Status

# fit the model with data
model1.fit(X_train,y_train.values.ravel())#y_train.values.ravel() because a 1d array is expected, not a column
print(y_train)
y_pred=model1.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
# print(cnf_matrix)

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
# plt.show()
train_acc = model1.score(X_train, y_train)
print("The Accuracy for Training Set is {}".format(train_acc*100))
test_acc = model1.score(X_test, y_test)
print("The Accuracy for Testing Set is {}".format(test_acc*100))
# scores = cr s_val_score(model1, X_train, y_train.values.ravel(), cv=10)
# print('Cross-Validation Accuracy Scores', scores)
