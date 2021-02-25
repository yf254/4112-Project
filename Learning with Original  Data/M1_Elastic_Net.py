# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 16:18:09 2021
Credit Risk Analysis using machine learning and deep learning models
Machine Learning Model 1: Elastic Net
@author: Feng Yuxiao 1155107773
"""

import pandas as pd
import matplotlib.pyplot as plt
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
col_num = X.shape[1]
print(col_num)
y = all_data['DEFAULT']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=100)
X_train,X_valid,y_train,y_valid = train_test_split(X_train,y_train,test_size=0.25,random_state=100)

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
plot_roc(y_valid, y_hat1,'M1','valid data')

# Testing Set
plot_roc(y_test, y_hat2,'M1','test data')