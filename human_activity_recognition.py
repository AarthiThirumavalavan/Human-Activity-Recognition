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
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

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
model.add(Dense(64, activation='relu', input_dim=561))
model.add(Dropout(0.5))#Dropout: used to set a fraction rate of input units to 0 at each update during training time which helps prevent overfitting
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(train_data, train_label_O,
          epochs=150,
          batch_size=128)

score_NN = model.evaluate(test_data, test_label_O, batch_size=128)
print("Model score for Keras Neural Network:{:.3f}".format(score_NN[1]))

###K-NEAREST NEIGHBORS
knn = KNeighborsClassifier(n_neighbors=20 , n_jobs=2 , weights='distance')
knn.fit(train_data, train_label)
score_knn = knn.score(test_data,test_label)
print("Model score for K-Nearest Neighbors:{:.3f}".format(score_knn))

###SUPPORT VECTOR MACHINES
#param_grid={'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}
param_grid={'C':[0.1,1],'gamma':[1,0.1]}
scaling = MinMaxScaler(feature_range=(-1,1)).fit(train_data)
X_train = scaling.transform(train_data)
X_test = scaling.transform(test_data)
grid=GridSearchCV(SVC(),param_grid,scoring="accuracy")
grid.fit(X_train, train_label)
print(grid.best_params_)
grid_predictions = grid.predict(test_data)
score_SVM = metrics.accuracy_score(test_label, grid_predictions)
print(classification_report(test_label, grid_predictions))
print("Accuracy for Decision Tree: {:.3f}".format(metrics.accuracy_score(test_label, grid_predictions)))

###DECISION TREE
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(train_data, train_label)
score_decision = decision_tree.score(test_data,test_label)
print("Accuracy for Decision Tree: {:.3f}".format(score_decision))

###GRADIENT BOOSTED DECISION TREE
gb = GradientBoostingClassifier(learning_rate = 0.01, max_depth=2, random_state=0)
gb.fit(train_data, train_label)
score_gbdt = gb.score(test_data,test_label)
print('Accuracy of GBDT classifier on test set: {:.3f}'.format(score_gbdt))

###RANDOM FORESTS
randomforest = RandomForestRegressor(n_estimators = 10, max_depth = 6, min_samples_leaf = 10, n_jobs = 4)
randomforest.fit(train_data, train_label)
score_rf = randomforest.score(test_data, test_label)
print('Accuracy of Random Forest classifier on test set: {:.3f}'.format(score_rf))

###LOGISTIC REGRESSION
logistic = LogisticRegression(C=1)
logistic.fit(train_data, train_label)
score_logistic = logistic.score(test_data, test_label)
print('Accuracy of Logistic Regression on test set: {:.3f}'.format(score_logistic))

###GAUSSIAN NAIVE BAYES CLASSIFIER
gaussian_nb = GaussianNB()
gaussian_nb.fit(train_data, train_label)
score_gaussian_nb = gaussian_nb.score(test_data, test_label)
print('Accuracy of Gaussian Naive Bayes on test set: {:.3f}'.format(score_gaussian_nb))

###SINGLE LAYER PERCEPTRON
score_single_MLP = []
for units in ([1,10,25,100]):
        single_MLP = MLPClassifier(hidden_layer_sizes=[units], solver='lbfgs', random_state=0)
        single_MLP.fit(train_data, train_label)
        score = single_MLP.score(test_data, test_label)
        score_single_MLP.append(score)
maxscore = max(score_single_MLP)
print('Maximum accuracy of Single Layer Perceptron on test set: {:.3f}'.format(maxscore))
        
###MULTILAYER PERCEPTRON
MLP = MLPClassifier(hidden_layer_sizes = [10, 100], solver='lbfgs',random_state = 0)
MLP.fit(train_data, train_label)        
score_MLP = MLP.score(test_data, test_label) 
print('Accuracy of Multiple Layer Perceptron on test set: {:.3f}'.format(score_MLP))

###CONSOLIDATED REPORT
consolidated_score = [{'ALGORITHM': 'Keras Neural Network', 'SCORE': score_NN[1]},
         {'ALGORITHM': 'Multilayer Perceptron', 'SCORE': score_MLP},
         {'ALGORITHM': 'Decision Tree', 'SCORE': score_decision },
         {'ALGORITHM': 'Gaussian Naive Bayes', 'SCORE': score_gaussian_nb },
         {'ALGORITHM': 'Gradient Boosted Decision tree', 'SCORE': score_gbdt },
         {'ALGORITHM': 'K Nearest Neighbor', 'SCORE': score_knn },
         {'ALGORITHM': 'Logistic Regression', 'SCORE': score_logistic },
         {'ALGORITHM': 'Random Forests', 'SCORE': score_rf },
         {'ALGORITHM': 'Single MLP', 'SCORE': maxscore },
         {'ALGORITHM': 'SVM', 'SCORE': score_SVM}]
df = pd.DataFrame(consolidated_score)
print(df)
df['SCORE'].idxmax()
print("Maximum Accuracy model")
print(df.loc[df['SCORE'].idxmax()])
