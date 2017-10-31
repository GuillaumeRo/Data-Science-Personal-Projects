# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 11:05:26 2017

@author: station
"""
########################### Comments ##########################################
#Removing outliers in a very barbaric way
#Imputing missing values when it is easy to guess
#Removing columns for the other missing values keeping MSZoning>num
#Transforming ordered categorical variables into numerical variables
#Correcting MSSubClass
#Create polynomial variables + new +fined tuned
#Create dummy variables
#zscoring on all_data (-0.3 RMSE)
#Better lasso cross validation
#Checking most important variables in the model
########################### Load libraries#####################################
print("### Loading Libraries ###")

#datetime
import datetime
#chdir
import os
start_time = datetime.datetime.now()

#Regex
import re

# pandas
import pandas as pd
from pandas import Series,DataFrame
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from scipy.stats import skew
from scipy.stats import boxcox
from scipy.stats import boxcox_normmax
import xgboost as xgb
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn import datasets, linear_model
import datetime
from sklearn.metrics import mean_squared_error, make_scorer
print("Done in", datetime.datetime.now()-start_time)

########################## Load Dataset #######################################
print("### Loading Data ###")
start_time = datetime.datetime.now()

#Set working directory
os.chdir("D:\\Users\\station\\Documents\\GitHub\\Data-Science-Personal-Projects\\House Prices Advanced Regression Techniques")

#Load train data
train = pd.read_csv("Data\\train.csv")
#Drop Id column in train
train=train.drop("Id",axis=1)


#Load test data
test=pd.read_csv("Data\\test.csv")
#Save Id column separately in test_Id and remove it
test_Id=test['Id']
test=test.drop("Id",axis=1)

print("Done in", datetime.datetime.now()-start_time)

####################### Removing Outliers #####################################
print("### Removing outliers ###")

# outlier deletion
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
train = train.drop(train[(train['TotalBsmtSF']>4000) & (train['SalePrice']<300000)].index)

#Save SalePrice column separately in train_SalePrice and remove it from train
SalePrice=train['SalePrice']
train=train.drop("SalePrice", axis=1)

print("Done in", datetime.datetime.now()-start_time)
####################### Missing values ########################################

print("### Missing Values ###")
start_time = datetime.datetime.now()

#First, impute the missing value that we can guess with a 100% accuracy, Then remove the other
#Check the number of missing values in train and test
MissingValues_train_Row=train.isnull().sum(axis=1)
MissingValues_test_Row=test.isnull().sum(axis=1)

MissingValues_train_Col=train.isnull().sum()
MissingValues_test_Col=test.isnull().sum()


#Alley: #cf Data description: NAs mean "No alley access" ==> Impute "No" value
train['Alley']=train['Alley'].fillna("No")
test['Alley']=test['Alley'].fillna("No")

#BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2
#Data Description: NAs mean non basement==> Impute "NoB" value
for c in ('BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2'):
    train[c]=train[c].fillna("NoB")
    test[c]=test[c].fillna("NoB")
#Make sure that if no basement, then all missing measures related to basement are set to zero
for c in ('BsmtHalfBath','BsmtFullBath','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFinSF1'):
    train[c]=train[c].fillna(0)
    test[c]=test[c].fillna(0)  

#Electrical
Electrical_train_Null=train.loc[train['Electrical'].isnull()]
Electrical_test_Null=test.loc[test['Electrical'].isnull()]
#This observation in train has Central Air Conditioning, so it has electricity
#Year Built is 2006 and all houses built after 1966 have "Standard Circuit Breakers & Romex"
#Replace NAs by "SBrkr"
train['Electrical']=train['Electrical'].fillna("SBrkr")

#Fence: Fence quality. According to the data description, NAs mean no fence, so we replace NAs by "No"
train['Fence']=train['Fence'].fillna("No")
test['Fence']=test['Fence'].fillna("No")

#FireplaceQu: Fireplace quality, according to Data Description, NAs mean no fireplace 
#==>Replace NAs by "No"
train['FireplaceQu']=train['FireplaceQu'].fillna("No")
test['FireplaceQu']=test['FireplaceQu'].fillna("No")

#GarageType, GarageYrBlt, GarageFinish, GarageQual, GarageCond. 
#According to Data Description, NAs mean no garage==>Replace NAs by "No"
for c in ('GarageType','GarageFinish','GarageQual','GarageCond'):
    train[c]=train[c].fillna("No")
    test[c]=test[c].fillna("No")

#GarageArea, GarageCars
#Missing values actually have no garage ==>0
#Impute value
for c in ('GarageCars','GarageArea'):
    train[c]=train[c].fillna(0)
    test[c]=test[c].fillna(0)

#MiscFeature: Miscellaneous feature not covered in other categories
#According to the data description, NAs mean "None"
train['MiscFeature']=train['MiscFeature'].fillna("No")
test['MiscFeature']=test['MiscFeature'].fillna("No")

#PoolQC: Pool quality: According to the data description, NAs mean no pool, so we replace NAs by "No"
train['PoolQC']=train['PoolQC'].fillna("No")
test['PoolQC']=test['PoolQC'].fillna("No")

#GarageYrBlt
train['GarageYrBlt']=train['GarageYrBlt'].fillna(0)
test['GarageYrBlt']=test['GarageYrBlt'].fillna(0)

#KitchenQual
#Replace by the most frequent value
test['KitchenQual']=test['KitchenQual'].fillna(test['KitchenQual'].value_counts().index[0])

#Keeping MSZoning as it is a geographical variable
#Filling missing values with the most frequent value
test['MSZoning']=test['MSZoning'].fillna(test['MSZoning'].value_counts().index[0])
train['MSZoning']=train['MSZoning'].fillna(train['MSZoning'].value_counts().index[0])



# search for missing data
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_train = pd.concat([total, percent], axis=1, keys=['Total','Percent'])
total = test.isnull().sum().sort_values(ascending=False)
percent = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)
missing_test = pd.concat([total, percent], axis=1, keys=['Total','Percent'])


# delete all the columns with missing data
outidx = list(missing_train[missing_train['Total']>1].index) + \
    list(missing_test[missing_test['Total']>=1].index)
train = train.drop(outidx, 1)
test = test.drop(outidx, 1)

print("Done in", datetime.datetime.now()-start_time)

#################### Replacing ordered categorical variables  by numbers #####
#KitchenQual: Kitchen quality
#
#       Ex	Excellent
#       Gd	Good
#       TA	Typical/Average
#       Fa	Fair
#       Po	Poor
di = {"Po": 1, "Fa": 2, "TA":3, "Gd":4, "Ex":5}
train['KitchenQual'].replace(di,inplace=True)
test['KitchenQual'].replace(di,inplace=True)


#PavedDrive: Paved driveway
#
#       Y	Paved 
#       P	Partial Pavement
#       N	Dirt/Gravel
di = {"N": 1, "P":2, "Y":3}
train['PavedDrive'].replace(di,inplace=True)
test['PavedDrive'].replace(di,inplace=True)

##ExterQual: Evaluates the quality of the material on the exterior 
##		
##       Ex	Excellent
##       Gd	Good
##       TA	Average/Typical
##       Fa	Fair
##       Po	Poor
#di = {"Po": 1, "Fa": 2, "TA":3, "Gd":4, "Ex":5}
#train['ExterQual'].replace(di,inplace=True)
#test['ExterQual'].replace(di,inplace=True)

#Does not improve rmse
#LotShape: General shape of property
#"Reg", IR1, IR2 and IR3 can be replaced by 0, 1, 2 and 3, descriibing 
#the degre of irregularity of the shape of the property
#di = {"Reg": 0, "IR1": 1, "IR2":2, "IR3":3}
#train['LotShape'].replace(di,inplace=True)
#test['LotShape'].replace(di,inplace=True)

##LandContour: Flatness of the property
##Lvl, Bnk, HLS and Low can be replaced by 0, 1, 2 and 3, describing
##the degre of "non-flatness" of the property
#di = {"Lvl": 0, "Bnk": 1, "HLS":2, "Low":3}
#train['LandContour'].replace(di,inplace=True)
#test['LandContour'].replace(di,inplace=True)

##ExterCond: Evaluates the present condition of the material on the exterior
##		
##       Ex	Excellent
##       Gd	Good
##       TA	Average/Typical
##       Fa	Fair
##       Po	Poor
#di = {"Po": 1, "Fa": 2, "TA":3, "Gd":4, "Ex":5}
#train['ExterCond'].replace(di,inplace=True)
#test['ExterCond'].replace(di,inplace=True)

#Does not improve rmse
#BsmtQual: Evaluates the height of the basement
#
#       Ex	Excellent (100+ inches)	
#       Gd	Good (90-99 inches)
#       TA	Typical (80-89 inches)
#       Fa	Fair (70-79 inches)
#       Po	Poor (<70 inches
#       NoB	No Basement
#di = {"NoB": 0, "Po": 1, "Fa":2, "TA":3, "Gd":4, "Ex":5}
#train['BsmtQual'].replace(di,inplace=True)
#test['BsmtQual'].replace(di,inplace=True)
#		


#Variable removing because of missing values
#Utilities: Type of utilities available
#AllPub	All public Utilities (E,G,W,& S)	
#NoSewr	Electricity, Gas, and Water (Septic Tank)
#NoSeWa	Electricity and Gas Only
#ELO	Electricity only
#di = {"ELO": 1, "NoSeWa": 2, "NoSewr":3, "AllPub":4}
#train['Utilities'].replace(di,inplace=True)
#test['Utilities'].replace(di,inplace=True)

#Does not improve rmse
#LandSlope: Slope of property
#		
#       Gtl	Gentle slope
#       Mod	Moderate Slope	
#       Sev	Severe Slope
#di = {"Gtl": 1, "Mod": 2, "Sev":3}
#train['LandSlope'].replace(di,inplace=True)
#test['LandSlope'].replace(di,inplace=True)

###### Correcting MSSubclass: replacing numbers by categories ################
di = {20:"SC20", 30:"SC30", 40:"SC40", 45:"SC45", 50:"SC50", 60:"SC60", 70:"SC70", 75:"SC75", 80:"SC80", 85:"SC85", 90:"SC90", 120:"SC120", 150:"SC150", 160:"SC160", 180:"SC180", 190:"SC190"}
train['MSSubClass'].replace(di,inplace=True)
test['MSSubClass'].replace(di,inplace=True)

############# Grouping train and test data for further processing #############
train['istrain']=1
test['istrain']=0

all_data = pd.concat([train,test])
all_data_istrain=all_data['istrain']
all_data=all_data.drop('istrain',axis=1)

##################### Creating new variables #################################
#Create polynomial features for the most correlated variables

all_data['OverallQual1/2']=all_data['OverallQual']**(1/2)
all_data['OverallQual2']=all_data['OverallQual']**2
all_data['OverallQual3']=all_data['OverallQual']**3

all_data['OverallCond1/2']=all_data['OverallCond']**(1/2)
all_data['OverallCond2']=all_data['OverallCond']**2
all_data['OverallCond3']=all_data['OverallCond']**3


all_data['LotArea1/2']=all_data['LotArea']**(1/2)
all_data['LotArea2']=all_data['LotArea']**2
all_data['LotArea3']=all_data['LotArea']**3


all_data['GrLivArea1/9']=all_data['GrLivArea']**(1/9)
#Not improving rmse
#all_data['GrLivArea1/8']=all_data['GrLivArea']**(1/8)
#all_data['GrLivArea1/7']=all_data['GrLivArea']**(1/7)
#all_data['GrLivArea1/6']=all_data['GrLivArea']**(1/6)
#all_data['GrLivArea1/5']=all_data['GrLivArea']**(1/5)
#all_data['GrLivArea1/4']=all_data['GrLivArea']**(1/4)
#all_data['GrLivArea1/3']=all_data['GrLivArea']**(1/3)
#all_data['GrLivArea1/2']=all_data['GrLivArea']**(1/2)
#all_data['GrLivArea2']=all_data['GrLivArea']**2
all_data['GrLivArea3']=all_data['GrLivArea']**3

#Not improving rmse
#all_data['YearBuilt1/2']=all_data['YearBuilt']**(1/2)
#all_data['YearBuilt2']=all_data['YearBuilt']**2
all_data['YearBuilt3']=all_data['YearBuilt']**3


#Does not improve rmse
#all_data['ExterQual1/2']=all_data['ExterQual']**(1/2)
#all_data['ExterQual2']=all_data['ExterQual']**2
#all_data['ExterQual3']=all_data['ExterQual']**3

#Not improving rmse
#all_data['KitchenQual11']=all_data['KitchenQual']**(11)
#all_data['KitchenQual7']=all_data['KitchenQual']**7
#all_data['KitchenQual8']=all_data['KitchenQual']**8
#all_data['KitchenQual9']=all_data['KitchenQual']**9
#all_data['KitchenQual10']=all_data['KitchenQual']**10
all_data['KitchenQual6']=all_data['KitchenQual']**6


all_data['GarageCars1/2']=all_data['GarageCars']**(1/2)
#Not improving rmse
#all_data['GarageCars2']=all_data['GarageCars']**2
#all_data['GarageCars3']=all_data['GarageCars']**3

#Not improving rmse
#all_data['GarageArea1/2']=all_data['GarageArea']**(1/2)
#all_data['GarageArea2']=all_data['GarageArea']**2
#all_data['GarageArea3']=all_data['GarageArea']**3

#Not improving rmse
#all_data['TotalBsmtSF1/2']=all_data['TotalBsmtSF']**(1/2)
#all_data['TotalBsmtSF2']=all_data['TotalBsmtSF']**2
#all_data['TotalBsmtSF3']=all_data['TotalBsmtSF']**3

all_data['1stFlrSF1/2']=all_data['1stFlrSF']**(1/2)
all_data['1stFlrSF2']=all_data['1stFlrSF']**2
all_data['1stFlrSF3']=all_data['1stFlrSF']**3

#################### Focusing on location variables ###########################
#MSZoning FV>RL>RH>RM>C(all)
di = {"FV":5, "RL":4, "RH":3, "RM":2, "C (all)":1}
all_data['MSZoning'].replace(di,inplace=True)


#Neighborhood
#Condition1
#Condition1

#################### Create dummy variables ###################################
print("### Creating dummy variables ###")
start_time = datetime.datetime.now()




for c in list(all_data):
    if all_data[c].dtypes=="object":
        all_data_dummy=pd.get_dummies(all_data[c],prefix=c)
        all_data.drop(c, inplace=True, axis=1)
        all_data = pd.concat([all_data, all_data_dummy], axis=1)
        

################## Standardization ############################################
all_data = (all_data - all_data.mean())/all_data.std()

all_data['istrain']=all_data_istrain

train=all_data.loc[all_data['istrain']==1]
test=all_data.loc[all_data['istrain']==0]

MissingValues_all_data_Col=all_data.isnull().sum()
MissingValues_train_Col=train.isnull().sum()
MissingValues_test_Col=test.isnull().sum()

print("Done in", datetime.datetime.now()-start_time)

#################### Log transform the Saleprice ##############################
SalePrice=np.log(SalePrice)

######################### Build models ########################################
# Define error measure for official scoring : RMSE
scorer = make_scorer(mean_squared_error, greater_is_better = False)

def rmse_cv_train(model):
    rmse= np.sqrt(-cross_val_score(model, train, SalePrice, scoring = scorer, cv = 10))
    return(rmse)

######################### Lasso linear regression##############################
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV

from sklearn import linear_model

# look for the best regularization term


lasso = LassoCV(alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 
                          0.3, 0.6, 1], 
                max_iter = 50000, cv = 10)

lasso.fit(train, SalePrice)
alpha = lasso.alpha_
print("Best alpha :", alpha)
lasso = LassoCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, 
                          alpha * .85, alpha * .9, alpha * .95, alpha, alpha * 1.05, 
                          alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, alpha * 1.35, 
                          alpha * 1.4], 
                max_iter = 50000, cv = 10)
lr=lasso.fit(train, SalePrice)
alpha = lasso.alpha_
print("Best alpha :", alpha)

print("Lasso RMSE on Training set :", rmse_cv_train(lasso).mean())
#print("Lasso RMSE on Test set :", rmse_cv_test(lasso).mean())
#Old: 0.114689652365
#New: 0.113789336072
#New: 0.113111192303 with polynomial
#v15: 0.113036224551
#v16: 0.112883713497
#v17: 0.112554061388
#v18: 0.111409199535
#v28: 0.110415866011
pred = lr.predict(test)

pred = np.exp(pred)

pred_df = pd.DataFrame(pred, index=test_Id, columns=['SalePrice'])
pred_df.to_csv('LinearLasso_Simplifiedv28_output.csv', header=True, index_label='Id')

######################### Model analysis ######################################
# Plot important coefficients
coefs = pd.Series(abs(lr.coef_), index =train.columns)
print("Lasso picked " + str(sum(coefs != 0)) + " features and eliminated the other " +  \
      str(sum(coefs == 0)) + " features")
imp_coefs = pd.concat([coefs.sort_values().head(10),
                     coefs.sort_values().tail(10)])
imp_coefs.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")
plt.show()

#Most important coefficients
#GrLivArea1/2 0.1252958306334802
#YearBuilt 0.05284044836873775
#OverallCond 0.04450026121023658
#TotalBsmtSF 0.043832635743867275
#OverallQual1/2 0.040011287760048295
#OverallQual3 0.03918975348166529
#BsmtFinSF1 0.02688600574640097
#Neighborhood_Crawfor 0.02002780775866949
#GarageCars1/2 0.018321753284147563
#LotArea 0.018261613835977658
#SaleCondition_Abnorml 0.01735635139258414
#KitchenQual3 0.01728070450846573
#GarageArea 0.0154741516104221

#
