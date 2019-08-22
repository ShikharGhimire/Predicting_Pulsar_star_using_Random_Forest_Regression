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
                          

#Creating an artificial neural networks to see how Neural network works
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

#Implementing Artificial Neural network
classifier = Sequential()
    classifier.add(Dense(output_dim = 5, init = 'uniform',activation = 'relu',input_dim = 8))#First hidden layer
    classifier.add(Dense(output_dim = 5, init = 'uniform',activation = 'relu' )) #Second hidden layer
    classifier.add(Dense(output_dim = 1, init = 'uniform',activation = 'sigmoid')) #Output layer. Sigmoid function for the last layer since the dependent variable is binary
    #Compiling the ANN
    classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])

#Using grid_search to find the best hyperparameters
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(output_dim = 5, init = 'uniform',activation = 'relu',input_dim = 8))#First hidden layer
    classifier.add(Dense(output_dim = 5, init = 'uniform',activation = 'relu' )) #Second hidden layer
    classifier.add(Dense(output_dim = 1, init = 'uniform',activation = 'sigmoid')) #Output layer. Sigmoid function for the last layer since the dependent variable is binary
    #Compiling the ANN
    classifier.compile(optimizer = optimizer,loss = 'binary_crossentropy',metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)
#Using grid search to find the best hyperparameters
parameters = {'batch_size':[5,10,15,20,30,40,50,60,70],
              'epochs':[20,30,40,50,60,70,80,90,100,200,300,400,500],
              'optimizer':['adam','rmsprop']}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

grid_search = grid_search.fit(X_train,y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
