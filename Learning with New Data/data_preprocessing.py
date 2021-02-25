# -*- coding: utf-8 -*-
"""
Credit Risk Analysis using machine learning and deep learning models
New Data: Preprocessing
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
from sklearn.ensemble import RandomForestClassifier

def deal_with_na(df):
    pct_null = df.isnull().sum()/len(df) 
    missing_features = pct_null[pct_null > 0.40].index 
    df.drop(missing_features, axis=1, inplace=True) 
    all_features = [col for col in df.columns if col not in ['TARGET']]
    cat = [col for col in all_features if df[col].dtype == 'O']
    cont = [col for col in all_features if col not in cat]
    imputer1 = SimpleImputer(strategy='mean')
    df[cont] = imputer1.fit_transform(df[cont])
    imputer2 = SimpleImputer(strategy='most_frequent')
    df[cat] = imputer2.fit_transform(df[cat])
    return df

def get_dummies(df):
    all_features = [col for col in df.columns if col not in ['TARGET']]
    cat = [col for col in all_features if df[col].dtype == 'O']
    df = pd.get_dummies(df, columns=cat,drop_first=True)
    return df

def preprocess(df):
    deal_with_na(df)
    df = get_dummies(df)
    return df

#------------------Data Cleaning-----------------#
all_data=pd.read_csv("kaggle_application_train.csv")
all_data = preprocess(all_data)

#all_data.dropna(axis=1, how='all')
same = all_data.columns[all_data.nunique()==1]
all_data = all_data.drop(columns=same)
all_data.to_csv("new_data_preprocessed.csv")

