# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 15:33:38 2018

@author: Ashish
"""
"""
------------A CAVEAT--------------

There are assumptions while doing a linear regression. The assumptions are:
    
 1) Linearity
 2) Homoscedasticity
 3) Multivariate normality
 4) Lack of multicollinearity
 5) Independence of errors

Make sure to fulfill all the 5 requirements to form a linear regression
"""

"""
Technically there are five ways of building models:
 1) All- in: Put all variables into the model(only do it if we have prior knowledge regarding the variables)
    
 2) Backward Elimination [STEPWISE REGRESSION]-------------{The fastest one among all}
      a) Select a significance level to stay in the model(SL).
      b) Fit the full model with all possible predictors
      c) Consider the predictor with the highest P-value. If P > SL, go to step d otherwise Finish.
      d) Remove the predictor with P-values higher than significance level
      e) Fit model without this variable
      f) Repeat step c,d,e until all the variables left has P-values is less than significance levels. Model is ready one its achieved.                                    
 
 3) Forward Selection [STEPWISE REGRESSION]
      a) Select a significance level to enter the model(SL).
      b) Fit all simple regression models Y ~ Xn and select the one with the lowest P-value
      c) Keep this variable selected in step (b) and fit all possible models with one extra predictor added to the one(s) you already have.
      d) Consider the predictor with the lowest P-value. If P < SL, go to step c otherwise Finish.  
     
 4) Bidirectional Elimination [STEPWISE REGRESSION]
     a) Select a significance level to stay or to enter the model (This will have two SLs---> SL(E) and SL(S))
     b) Fit all simple regression models Y ~ Xn and select the one with the lowest P-value(new variables must have: P < SL(E) to enter))
     c) Perform all the steps of Backward Elimination(old variables msut have P-value < SL(S)to stay)
     d) No new variable to enter or exit. The model is ready.
    
 5) Score Comparison
     a) Select a criterion of goodness of fit(e.g Akaike criterion, R-square, etc)
     b) Construct all possible regression models: 2^n-1 total combinations
     c) Select the one with the best criterion

"""


#Importing the libraries
import numpy as np #Contains mathematical tools and used to include any type of mathematical codes
import matplotlib.pyplot as plt #This allows to plot charts 
import pandas as pd #Import and manage datasets


#Import the datasets using pandas 
dataset = pd.read_csv ('50_Startups.csv')

#Matrix of features. In this X is the independent data and Y is the dependent data.
X = dataset.iloc[:, :-1].values #This line means that we are taking all the columns except the last ome 
Y = dataset.iloc[:,4].values #This line means that we are taking the last column of that is "Purchased" 
"""
#How to handle missing datas.
#Solution: Replacing the missing data with the mean of all the data within that column.

from sklearn.preprocessing import Imputer # Imputer class allows us to take care of the missing datas
#All the columns contains NaN will be considered as missing value and will be replaced
imputer = Imputer(missing_values = 'NaN', strategy ='mean', axis = 0) 
#fitting the imputer obj. to matrix X
imputer = imputer.fit(X[:,1:3])
# Replace the missing data with the mean of the column in matrix X. 
X[:, 1:3] = imputer.transform(X[:, 1:3]) 

"""
#Encoding categorical data

from sklearn.preprocessing import LabelEncoder # LabelEncoder class from sklearn.preprocessor library used to encode the values
label_encoder_X = LabelEncoder()
X[:, 3] = label_encoder_X.fit_transform(X[:, 3]) # fitted label_encoded_X obj to the first column country. It will no longer consist of names of the counteries.

#Dummy encodings. Making the countries names into 0-california, 1- florida and 2-NY.

from sklearn.preprocessing import OneHotEncoder # this is used to create the dummy variables
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the dummy variable trap
 
X = X[:, 1:] # removed first column from X and taking all the column of X from 1st index till the end.

#Training and Testing dataset split

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

# Fitting multiple linear regression to the training set

from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(X_train, Y_train)

#Predicting the Test set results
#Create a vector containing the prediction of the salary

Y_pred = linear_reg.predict(X_test)

# ----------------BACKWARD ELIMINATION-------------------
import statsmodels.formula.api as sm  #  
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1) #adding the columns of 1s in X due to the requirement of the library used

X_opt = X [:, [0, 1, 2, 3, 4, 5]] 

reg_OLS = sm.OLS(endog = Y, exog = X_opt).fit() 
reg_OLS.summary()  
#--------------------------------------------------
X_opt = X [:, [0, 1, 3, 4, 5]] 

reg_OLS = sm.OLS(endog = Y, exog = X_opt).fit() 
reg_OLS.summary() 
#--------------------------------------------------
X_opt = X [:, [0, 3, 4, 5]] 

reg_OLS = sm.OLS(endog = Y, exog = X_opt).fit() 
reg_OLS.summary() 
#--------------------------------------------------
X_opt = X [:, [0, 3, 5]] 

reg_OLS = sm.OLS(endog = Y, exog = X_opt).fit() 
reg_OLS.summary() 
#--------------------------------------------------
X_opt = X [:, [0, 3]] 

reg_OLS = sm.OLS(endog = Y, exog = X_opt).fit() 
reg_OLS.summary() 
#--------------------------------------------------
X_opt = X [:, [3]] 

reg_OLS = sm.OLS(endog = Y, exog = X_opt).fit() 
reg_OLS.summary() 
#--------------------------------------------------

# Dummy encoding for Y "Profit" column
label_encoder_Y = LabelEncoder()
Y = label_encoder_X.fit_transform(Y) # fitted label_encoded_Y obj to the column Y "profit". It will no longer consist of YES = 1 or NO = 0.


#Feature scaling is used to avoid the highly huge or small number to avoid domination of one variable over another. It could be done either with standarisation or normalization

from sklearn.preprocessing import StandardScaler
ss_X = StandardScaler()
X_train = ss_X.fit_transform(X_train) #applying ss_X to the training set, then the data must first be fit then trained 
X_test = ss_X.transform(X_test) #applying ss_X to the testing set, then the data only need to be transformed.