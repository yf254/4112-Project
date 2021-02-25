# -*- coding: utf-8 -*-
"""
Credit Risk Analysis using machine learning and deep learning models
Ten Feature Analysis: EN
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score,roc_curve,auc

#------------------Data Cleaning-----------------#
all_data=pd.read_csv("new_data_preprocessed.csv")
X = all_data.drop(columns=['TARGET'])
y = all_data['TARGET']

#------------------Ten Feature-------------------#
X_m1 = X.iloc[:,[34,11,8,10,5,4,9,172,57,64]]
X_m2 = X.iloc[:,[28,29,30,79,26,60,9,2,36,16]]
X_m3 = X.iloc[:,[29,28,79,16,18,83,2,57,34,64]]
X_m31 = X.iloc[:,[80,81,56,79,27,11,82,2,18,16]]
X_m32 = X.iloc[:,[29,18,26,79,16,28,82,60,2,34]]
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

#---------------Model 1: Elastic Net (logic regression with regularization)---------#
from sklearn import linear_model
elastic= linear_model.ElasticNet(alpha=0.5)
elastic.fit(X_trains,y_trains)
# Validation Set
y_hat1 = elastic.predict(X_valid)  
print("M1(valid) RMSE:", np.sqrt(metrics.mean_squared_error(y_valid, y_hat1))) 
print("M1(valid) AUC:",roc_auc_score(y_valid, y_hat1))
# Testing Set
y_hat2 = elastic.predict(X_test)  
print("M1(test) RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_hat2))) 
print("M1(test) AUC:",roc_auc_score(y_test, y_hat2))



