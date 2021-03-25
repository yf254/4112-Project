"""
Credit Risk Analysis using machine learning and deep learning models
Sampling_Method_GBDT
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

#-------------------Stacking Model Building-----------------#
def build_model_gbdt(X_trains,y_trains):
    gbdt = GradientBoostingRegressor(n_estimators=127,max_depth=9,
                                     learning_rate=0.05,subsample=0.65,random_state=100)  #0.7156132998351604
    gbdt.fit(X_trains,y_trains)
    return gbdt

def build_model_gbm(X_trains,y_trains):
    gbm = GradientBoostingClassifier(n_estimators=11,learning_rate=0.05,subsample=0.5,
                                 max_depth=2,random_state=114)
    gbm.fit(X_trains,y_trains)
    return gbm

def build_model_en(X_trains,y_trains):
    elastic= linear_model.ElasticNet(alpha=0.05)
    elastic.fit(X_trains,y_trains)
    return elastic

def build_model_lr(X_trains,y_trains):
    reg_model = linear_model.LinearRegression()
    reg_model.fit(X_trains,y_trains)
    return reg_model

model_gbdt = build_model_gbdt(X_trains,y_trains)
pre1v_gbdt = model_gbdt.predict(X_valid)
pre1_gbdt = model_gbdt.predict(X_test)

model_gbm = build_model_gbm(X_trains,y_trains)
pre1_gbm = model_gbm.predict(X_test)
pre1v_gbm = model_gbm.predict(X_valid)

model_en = build_model_en(X_trains,y_trains)
pre1_en = model_en.predict(X_test)
pre1v_en = model_en.predict(X_valid)

# Level 1
train_gbdt_pred = model_gbdt.predict(X_trains)
train_gbm_pred = model_gbm.predict(X_trains)
train_en_pred = model_en.predict(X_trains)

Level1_X_train = pd.DataFrame()
Level1_X_train['gbdt'] = train_gbdt_pred
Level1_X_train['gbm'] = train_gbm_pred
Level1_X_train['en'] = train_en_pred

Level1_X_valid = pd.DataFrame()
Level1_X_valid['gbdt'] = pre1v_gbdt
Level1_X_valid['gbm'] = pre1v_gbm
Level1_X_valid['en'] = pre1v_en

Level1_X_test = pd.DataFrame()
Level1_X_test['gbdt'] = pre1_gbdt
Level1_X_test['gbm'] = pre1_gbm
Level1_X_test['en'] = pre1_en

# Level 2
Level2_model_lr = build_model_lr(Level1_X_train,y_trains)

Level2_train_pred = Level2_model_lr.predict(Level1_X_train)

y_hat1 = Level2_model_lr.predict(Level1_X_valid)
y_hat1 = np.around(y_hat1)
y_hat1 = list(map(abs,y_hat1))
y_hat1 = list(map(int,y_hat1))

y_hat2 = Level2_model_lr.predict(Level1_X_test)
y_hat2 = np.around(y_hat2)
y_hat2 = list(map(abs,y_hat2))
y_hat2 = list(map(int,y_hat2))

# Validation Set  
print("Stacking(valid) RMSE:", np.sqrt(metrics.mean_squared_error(y_valid, y_hat1))) 
print("Stacking(valid) AUC:",roc_auc_score(y_valid, y_hat1))
# Testing Set 
print("Stacking(test) RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_hat2))) 
print("Stacking(test) AUC:",roc_auc_score(y_test, y_hat2))

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
plot_roc(y_valid,y_hat1,'Stacking','valid data')

# Testing Set
plot_roc(y_test, y_hat2,'Stacking','test data')
