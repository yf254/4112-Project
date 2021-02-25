# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 16:18:09 2021
Credit Risk Analysis using machine learning and deep learning models
Deep Learning Model 4
@author: Feng Yuxiao 1155107773
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree   
import matplotlib.pyplot as plt
from keras.models import Sequential
from tensorflow import keras
import tensorflow as tf
from keras.layers import Dense
from keras.layers import Activation
from tensorflow.python.framework import ops
from keras.utils import to_categorical 
from keras import models 
from keras import layers
from keras import optimizers
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score,roc_curve,auc

#------------------Data Cleaning-----------------#
all_data=pd.read_csv("all_data.csv")
#all_data.dropna(axis=1, how='all')
same = all_data.columns[all_data.nunique()==1]
all_data = all_data.drop(columns=same)

X = all_data.drop(columns=['DEFAULT'])
y = all_data['DEFAULT']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=80)
X_train,X_valid,y_train,y_valid = train_test_split(X_train,y_train,test_size=0.25,random_state=80)

#------------------Data Preprosessing --------------------#

# SMOTE Algorithm to get balanced set
from collections import Counter
#print(Counter(y_train))
# It is an imbalanced dataset, so we need to use SMOTE algorithm to get it balanced
from imblearn.over_sampling import SMOTE
smo = SMOTE(random_state=99)
X_trains, y_trains = smo.fit_sample(X_train, y_train)
print(Counter(y_trains))

#---------------Deep Learning Model 4---------#
def build_D4():
    model = keras.Sequential([
    keras.layers.Dropout(0, input_shape=(106,)),
    keras.layers.Flatten(input_shape=(106,)),
    keras.layers.Dense(50, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.L1(6.81e-5),
    activity_regularizer=tf.keras.regularizers.L2(6.41e-5)),
    keras.layers.Dense(50, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.L1(6.81e-5),
    activity_regularizer=tf.keras.regularizers.L2(6.41e-5)),
    keras.layers.Dense(50, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.L1(6.81e-5),
    activity_regularizer=tf.keras.regularizers.L2(6.41e-5)),
    keras.layers.Dense(1, activation=tf.nn.sigmoid),
    ])
    model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
    return model

D4 = build_D4()
D4.fit(X_trains, y_trains)

# Validation Set
y_hat1 = D4.predict(X_valid)
y_hat1 = (y_hat1 > 0.5)
y_hat1 = y_hat1.astype(int)
y_hat1 = pd.DataFrame(y_hat1,columns=['DEFAULT'])
print("D4(valid) RMSE:", np.sqrt(metrics.mean_squared_error(y_valid, y_hat1))) 
print("D4(valid) AUC:",roc_auc_score(y_valid, y_hat1))

# Testing Set
y_hat2 = D4.predict(X_test)
y_hat2 = (y_hat2 > 0.5)
y_hat2 = y_hat2.astype(int)
y_hat2 = pd.DataFrame(y_hat2,columns=['DEFAULT'])
print("D4(test) RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_hat2))) 
print("D4(test) AUC:",roc_auc_score(y_test, y_hat2))

#---------------------ROC Curve-----------------------#
def plot_roc(labels, predict_prob,model_name,test_or_valid):
    false_positive_rate,true_positive_rate,thresholds=roc_curve(labels, predict_prob)
    roc_auc=auc(false_positive_rate, true_positive_rate)
    m = model_name
    t = test_or_valid
    plt.figure()
    plt.figure(figsize=(10,10))
    plt.title('%s: ROC Curve on %s for GLM' % (m,t))
    plt.plot(false_positive_rate, true_positive_rate,'b',label='AUC = %0.4f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.show()

# Validation Set
plot_roc(y_valid, y_hat1,'D4','valid data')

# Testing Set
plot_roc(y_test, y_hat2,'D4','test data')