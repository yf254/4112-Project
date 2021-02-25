# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 16:18:09 2021
Credit Risk Analysis using machine learning and deep learning models
New Data Deep Learning Model 1
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
from keras.utils import to_categorical 
from keras import models 
from keras import layers
from keras import optimizers
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score,roc_curve,auc
from sklearn.impute import SimpleImputer

#------------------Data Cleaning-----------------#
all_data=pd.read_csv("new_data_preprocessed.csv")
X = all_data.drop(columns=['TARGET'])
y = all_data['TARGET']
col_num = X.shape[1]
print(col_num)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=999)
X_train,X_valid,y_train,y_valid = train_test_split(X_train,y_train,test_size=0.25,random_state=999)

#------------------Data Preprosessing --------------------#

# SMOTE Algorithm to get balanced set
from collections import Counter
#print(Counter(y_train))
# It is an imbalanced dataset, so we need to use SMOTE algorithm to get it balanced
from imblearn.over_sampling import SMOTE
smo = SMOTE(random_state=99)
X_trains, y_trains = smo.fit_sample(X_train, y_train)
print(Counter(y_trains))

#---------------Deep Learning Model 1---------#
def build_D1():
    model = keras.Sequential([
    keras.layers.Flatten(input_shape=(col_num,)),
    keras.layers.Dense(int((2/3)*col_num), activation=tf.nn.tanh),
    keras.layers.Dense(int((2/3)*col_num), activation=tf.nn.tanh),
    keras.layers.Dense(1, activation=tf.nn.sigmoid),
    ])
    model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
    return model

D1 = build_D1()
D1.fit(X_trains, y_trains)

# Validation Set
y_hat1 = D1.predict(X_valid)
y_hat1 = (y_hat1 > 0.5)
y_hat1 = y_hat1.astype(int)
y_hat1 = pd.DataFrame(y_hat1,columns=['DEFAULT'])
print("D1(valid) RMSE:", np.sqrt(metrics.mean_squared_error(y_valid, y_hat1))) 
print("D1(valid) AUC:",roc_auc_score(y_valid, y_hat1))

# Testing Set
y_hat2 = D1.predict(X_test)
y_hat2 = (y_hat2 > 0.5)
y_hat2 = y_hat2.astype(int)
y_hat2 = pd.DataFrame(y_hat2,columns=['DEFAULT'])
print("D1(test) RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_hat2))) 
print("D1(test) AUC:",roc_auc_score(y_test, y_hat2))

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
plot_roc(y_valid, y_hat1,'D1','valid data')

# Testing Set
plot_roc(y_test, y_hat2,'D1','test data')














