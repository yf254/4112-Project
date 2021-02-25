# -*- coding: utf-8 -*-
"""
Credit Risk Analysis using machine learning and deep learning models
Sampling_Method_GBM
@author: Feng Yuxiao 1155107773
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn import metrics
from sklearn.metrics import roc_auc_score,roc_curve,auc
import category_encoders as ce
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier

#------------------Data Cleaning-----------------#
all_data=pd.read_csv("new_data_preprocessed.csv")
X = all_data.drop(columns=['TARGET'])
y = all_data['TARGET']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=999)
X_train,X_valid,y_train,y_valid = train_test_split(X_train,y_train,test_size=0.25,random_state=999)

#------------------Data Preprosessing --------------------#

# SMOTE Algorithm to get balanced set
from collections import Counter
#print(Counter(y_train))
# It is an imbalanced dataset, so we need to use SMOTE algorithm to get it balanced

#-------------------Other Sampling Method------------------#
#def plot_2d_space(X, y, label='Classes'):   
#    colors = ['#1F77B4', '#FF7F0E']
#    markers = ['o', 's']
#    for l, c, m in zip(np.unique(y), colors, markers):
#        plt.scatter(
#            X[y==l, 1],
#            X[y==l, 0],
#            c=c, label=l, marker=m
#        )
#    plt.title(label)
#    plt.legend(loc='upper right')
#    plt.show()
#from sklearn.decomposition import PCA

#pca = PCA(n_components=2)
#X_train = pca.fit_transform(X_train)
#plot_2d_space(X_train, y_train, 'Imbalanced dataset (2 PCA components)')

#SMOTE
#from imblearn.over_sampling import SMOTE
#smo = SMOTE(random_state=99)
#X_trains, y_trains = smo.fit_sample(X_train, y_train)
#print(Counter(y_trains))

#Random Over Sampling
#from imblearn.over_sampling import RandomOverSampler
#rdm = RandomOverSampler(random_state=99)
#X_trains, y_trains = rdm.fit_sample(X_train, y_train)
#print(Counter(y_trains))
#plot_2d_space(X_trains, y_trains, 'Random over-sampling')

#Random Under Sampling
from imblearn.under_sampling import RandomUnderSampler
rdum = RandomUnderSampler(random_state=99)
X_trains, y_trains = rdum.fit_sample(X_train, y_train)
print(Counter(y_trains))
#plot_2d_space(X_trains, y_trains, 'Random under-sampling')

#---------------Model 3: Gradient Boosting---------#
# Parameter Tuning
# Here, we tune n_estimator, learning_rate and subsample(boosting parameters, that included in the paper)
# and also max_depth, the tree parameter

Score = []
# Boosting Parameters
#for n in range(10,30,1): 
#for lr in np.arange(0.05,1,0.05):
#for ss in np.arange(0.5,1,0.05):
#for rs in range(200,210,1):
# Tree parameter
#for md in range(1,9,1):
#for tes in [1]:
#    gbm = GradientBoostingClassifier(n_estimators=11,learning_rate=0.05,subsample=0.5,
#                                     max_depth=2,random_state=114) #114 0.6036108983481302
#    gbm.fit(X_trains,y_trains)
#    score = roc_auc_score(y_valid, gbm.predict(X_valid))
#    Score.append(float(score))
#print(Score)


gbm = GradientBoostingClassifier(n_estimators=11,learning_rate=0.05,subsample=0.5,
                                 max_depth=2,random_state=114)
gbm.fit(X_trains,y_trains)
# Validation Set
y_hat1 = gbm.predict(X_valid)  
print("M3(valid) RMSE:", np.sqrt(metrics.mean_squared_error(y_valid, y_hat1))) 
print("M3(valid) AUC:",roc_auc_score(y_valid, y_hat1))
# Testing Set
y_hat2 = gbm.predict(X_test)  
print("M3(test) RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_hat2))) 
print("M3(test) AUC:",roc_auc_score(y_test, y_hat2))
