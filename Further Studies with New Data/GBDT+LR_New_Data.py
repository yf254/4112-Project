"""
Credit Risk Analysis using machine learning and deep learning models
GBDT+LR
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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model


#------------------Data Cleaning-----------------#
all_data=pd.read_csv("new_data_preprocessed.csv")
X = all_data.drop(columns=['TARGET'])
y = all_data['TARGET']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=999)
X_train,X_valid,y_train,y_valid = train_test_split(X_train,y_train,test_size=0.25,random_state=999)

#------------------Data Preprosessing --------------------#

# SMOTE Algorithm to get balanced set
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
rdm = RandomOverSampler(random_state=99)
X_trains, y_trains = rdm.fit_sample(X_train, y_train)
print(Counter(y_trains))

#-------------------GBDT+LR Model Building-----------------#
gbdt = GradientBoostingRegressor(n_estimators=127,max_depth=9,
                                     learning_rate=0.05,subsample=0.65,random_state=100)  #0.7156132998351604
gbdt.fit(X_trains,y_trains)

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(gbdt.apply(X_trains))
new_feature_train=enc.transform(gbdt.apply(X_trains))
new_feature_train=new_feature_train.toarray()
new_train=np.concatenate([X_trains,new_feature_train],axis=1)

new_feature_valid=enc.transform(gbdt.apply(X_valid))
new_feature_valid=new_feature_valid.toarray()
new_valid=np.concatenate([X_valid,new_feature_valid],axis=1)

new_feature_test=enc.transform(gbdt.apply(X_test))
new_feature_test=new_feature_test.toarray()
new_test=np.concatenate([X_test,new_feature_test],axis=1)

lr = linear_model.Ridge(alpha=1.5)
lr.fit(new_train, y_trains)

y_hat1 = lr.predict(X_valid)  
print("GBDT+LR(valid) RMSE:", np.sqrt(metrics.mean_squared_error(y_valid, y_hat1))) 
print("GBDT+LR(valid) AUC:",roc_auc_score(y_valid, y_hat1))
# Testing Set
y_hat2 = lr.predict(X_test)  
print("GBDT+LR(test) RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_hat2))) 
print("GBDT+LR(test) AUC:",roc_auc_score(y_test, y_hat2))


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
plot_roc(y_valid,y_hat1,'GBDT+LR','valid data')

# Testing Set
plot_roc(y_test, y_hat2,'GBDT+LR','test data')
