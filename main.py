#main function at a high level is
# STEP 1 make the synthetic file (or download a real file)
# STEP 2 preprocess using exp1.py file on readinf_in_csv branch.
#      2b                 then follow it up with fix_after.py
# STEP 3 Inspect data (train models)
#      3a. logistic regression model. Train x's on y's
#      3b. decision tree model. Train x's on y's
#      3c. 2 layer fully connected NN. For this, will need to normalize all inputs. Then, train x's on y's
# STEP 4 Visualize Data


#~~~~~~~`STEP 1 Make/download Data ~~~~~~~~
#make synthetic data by calling exp1.py and saving as unprocessed_csv
unprocessed_csv = exp1.makeData()

#~~~~~~~`STEP 2 Preprocess ~~~~~~~~
# load data to dataframe by using fix_after
data = fix_after.process(unprocessed_csv)

#~~~~~~~`STEP 3 Analyze Data ~~~~~~~~
#assign data to x and y
X_numpy = data.drop("App_Number", axis=1)
X = X_numpy.drop("Approval_Status", axis=1)
y = data[['Approval_Status']]
#split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y)

#~~~~~~~`STEP 3a Logistic Regression ~~~~~~~~
#need to edit exp1_syntheticdata_logregmodel.py such that it returns a model
logreg_model = exp1_syntheticdata_logregmodel.logreg(data)
train_acc = exp1_syntheticdata_logregmodel.get_train_acc(model, X_train, y_train)
test_acc = exp1_syntheticdata_logregmodel.get_test_acc(model, X_test, y_test)

#~~~~~~~`STEP 3b Decision Tree ~~~~~~~~
from sklearn import tree
dectree = tree.DecisionTreeClassifier()
dectree = dectree.fit(X, y)

#~~~~~~~`STEP 3c NN ~~~~~~~~
from sklearn.preprocessing import MinMaxScaler
# create scaler
scaler = MinMaxScaler()
# fit scaler on data
scaler.fit(data)
# apply transform
normalizedX = scaler.transform(data)
#split normalized input and output into training and testing data
normalized_X_train, normalized_X_test, y_train, y_test = train_test_split(normalizedX, y)
nn = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(5, 2), random_state=1) #guessing on the hidden layer sizes. any intuition on this?
#training. argument is the input data, and the target labels
nn.fit(normalized_X_train, y_train)
#testing. predict labels. argument is the input data
nn.predict(normalized_X_test)
#testing predict probabilities (gauge confidence). argument is the input data
nn.predict_proba(normalized_X_test, y_test)

#~~~~~~~`STEP 4 Visualize ~~~~~~~~

#log_reg confusion matrix
exp1_syntheticdata_logregmodel.cnf_matrix(model,X_test,y_test)

#tree
tree.plot_tree(dectree)

#NN
