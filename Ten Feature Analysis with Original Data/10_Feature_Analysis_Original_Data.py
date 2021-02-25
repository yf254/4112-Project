"""
Created on Tue Jan 19 16:18:09 2021
Credit Risk Analysis using machine learning and deep learning models
Ten Feature Analysis: Original Data
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

X_m1 = X.iloc[:,[0,1,2,3,4,5,6,7,8,9]]
X_m2 = X.iloc[:,[10,4,1,0,8,11,12,13,14,15]]
X_m3 = X.iloc[:,[10,5,16,17,18,19,20,21,22,23]]
X_d1 = X.iloc[:,[24,5,25,26,6,27,28,0,29,30]]
X_d2 = X.iloc[:,[10,31,32,33,34,35,36,37,38,39]]
X_d3 = X.iloc[:,[40,41,42,24,43,44,45,46,47,48]]
X_d4 = X.iloc[:,[49,6,50,51,52,53,54,55,56,43]]


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=80)
X_train,X_valid,y_train,y_valid = train_test_split(X_train,y_train,test_size=0.25,random_state=80)