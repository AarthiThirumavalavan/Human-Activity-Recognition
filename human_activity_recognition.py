# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 11:00:57 2018

@author: hp
"""
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import metrics
#Reading the files
train = shuffle(pd.read_csv("E:/Python Exercises/Human activity recognition/train.csv"))
test = shuffle(pd.read_csv("E:/Python Exercises/Human activity recognition/test.csv"))
#Assigning Label fields
train_labels = train["Activity"]
test_labels = test["Activity"]
#Dropping label and Subject field from train and test set
train_data = train.drop(["subject", "Activity"], axis = 1)
test_data = test.drop(["subject", "Activity"], axis = 1)
#Label encoding
labels = LabelEncoder()
labels = labels.fit(["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"])
train_label = labels.transform(train_labels)
test_label = labels.transform(test_labels)
#One hot encoding and to_categorical
#One hot encoding and to_categorical both are for the same concept
#for converting the label field to a rowsx6columns for output of 6 classes
train_label_C = to_categorical(train_label)
train_label_O = pd.get_dummies(train_label)
test_label_C = to_categorical(test_label)
test_label_O = pd.get_dummies(test_label)
#Model building
###KERAS NEURAL NETWORK
model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, activation='relu', input_dim=561))
model.add(Dropout(0.5))#Dropout: used to set a fraction rate of input units to 0 at each update during training time which helps prevent overfitting
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))
#SGD: Stochastic Gradient Descent Optimizer: an optimizer is one of the two arguments used to compile a keras model
#lr: learning rate , decay: learning rate decay over each update, momentum: used to accelerate SGD, nesterov: Whether to apply Nesterov momentum
#Nesterov momentum: type of momentum(momentum: used to obtain faster convergence)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#here loss=categorical crossentropy is used for multiclass problems
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(train_data, train_label_O,
          epochs=60,
          batch_size=128)

score_NN = model.evaluate(test_data, test_label_O, batch_size=128)
print("Model score for Keras Neural Network:{:.3f}".format(score_NN[1]))

###K-NEAREST NEIGHBORS
knn = KNeighborsClassifier(n_neighbors = 6)
knn.fit(train_data, train_label)
score_knn = knn.score(test_data,test_label)
print("Model score for K-Nearest Neighbors:{:.3f}".format(score_knn))

###SUPPORT VECTOR MACHINES
from sklearn.model_selection import GridSearchCV
#param_grid={'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}
param_grid={'C':[0.1,1],'gamma':[1,0.1]}
grid=GridSearchCV(SVC(kernel = "rbf"),param_grid, scoring="accuracy")
grid.fit(train_data, train_label)
print(grid.best_params_)
grid_predictions = grid.predict(test_data)
score_SVM = metrics.accuracy_score(test_label, grid_predictions)

###DECISION TREE
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier().fit(train_data, train_label)
print("Accuracy for Decision Tree: {:.3f}".format(decision_tree(test_data, test_label)))

###Gradient Boosted Decision Tree
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(learning_rate = 0.01, max_depth=2, random_state=0)
gb.fit(train_data, train_label)
print('Accuracy of GBDT classifier on test set: {:.2f}'.format(gb.score(test_data, test_label)))