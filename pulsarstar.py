#Pulsar star

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('pulsar_stars.csv')
X = dataset.iloc[:,0:8].values
y = dataset.iloc[:,8].values

#Splitting the dataset into training and testing dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25, random_state = 0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#Using random forest classification method to the dataset
from sklearn.ensemble import RandomForestClassifier
regressor = RandomForestClassifier(n_estimators = 19,random_state = 0) #Using grid search we found out that 19 for the n_estimators is the best value
regressor.fit(X_train,y_train)

#Predicting the X_test
y_pred = regressor.predict(X_test)

#Confusion matrix to check the accuracy
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred) #75 incorrect prediction before grid search method. Accuracy didn't change even after finding the best hyperparameters using the grid search

#Using k-fold cross validation and grid search method to increase the accuracy
#Applying k fold cross validation

#Using 10 fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=regressor, X = X_train, y = y_train, cv = 10, n_jobs=-1)
mean = accuracies.mean() #Checking out the means
standard_deviation = accuracies.std() #Standard deviation 

#Applying gridsearch method to improve the model performance by finding the right hyperparameters
from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}]

grid_search = GridSearchCV(estimator=regressor,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train,y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_                    
