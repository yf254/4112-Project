"""
Credit Risk Analysis using machine learning and deep learning models
Ten Feature Analysis: D2
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
col_num = 10

#------------------Ten Feature-------------------#
X_m1 = X.iloc[:,[34,11,8,10,5,4,9,172,57,64]]
X_m2 = X.iloc[:,[28,29,30,79,26,60,20,2,36,16]]
X_m3 = X.iloc[:,[29,28,79,16,18,83,2,56,55,64]]
X_m31 = X.iloc[:,[80,81,56,79,27,36,82,2,18,16]]
X_m32 = X.iloc[:,[29,18,26,79,16,28,82,60,2,30]]
X_m33 = X.iloc[:,[28,29,8,34,10,11,0,5,9,7]]

X_train,X_test,y_train,y_test = train_test_split(X_m33,y,test_size=0.2,random_state=999)
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

#---------------Deep Learning Model 2--------#
def build_D2():
    model = keras.Sequential([
    keras.layers.Flatten(input_shape=(col_num,)),
    keras.layers.Dense(40, activation=tf.nn.tanh),
    keras.layers.Dense(40, activation=tf.nn.tanh),
    keras.layers.Dense(40, activation=tf.nn.tanh),    
    keras.layers.Dense(1, activation=tf.nn.sigmoid),
    ])
    model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
    return model

D2 = build_D2()
D2.fit(X_trains, y_trains)

# Validation Set
y_hat1 = D2.predict(X_valid)
y_hat1 = (y_hat1 > 0.5)
y_hat1 = y_hat1.astype(int)
y_hat1 = pd.DataFrame(y_hat1,columns=['DEFAULT'])
print("D2(valid) RMSE:", np.sqrt(metrics.mean_squared_error(y_valid, y_hat1))) 
print("D2(valid) AUC:",roc_auc_score(y_valid, y_hat1))

# Testing Set
y_hat2 = D2.predict(X_test)
y_hat2 = (y_hat2 > 0.5)
y_hat2 = y_hat2.astype(int)
y_hat2 = pd.DataFrame(y_hat2,columns=['DEFAULT'])
print("D2(test) RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_hat2))) 
print("D2(test) AUC:",roc_auc_score(y_test, y_hat2))

