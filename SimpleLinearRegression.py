# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 12:38:11 2018

@author: Ashish
"""

#Importing the libraries
import numpy as np #Contains mathematical tools and used to include any type of mathematical codes
import matplotlib.pyplot as plt #This allows to plot charts 
import pandas as pd #Import and manage datasets


#Import the datasets using pandas 
dataset = pd.read_csv("Salary_Data.csv")

#Matrix of features. In this X is the independent data that is "YearsExperience " and Y is the dependent data "Salary".
X = dataset.iloc[:, :-1].values #This line means that we are taking all the columns except the last one 
Y = dataset.iloc[:,-1].values #This line means that we are taking the last column of that is "Purchased" 


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

#Encoding categorical data

from sklearn.preprocessing import LabelEncoder # LabelEncoder class from sklearn.preprocessor library used to encode the values
label_encoder_X = LabelEncoder()
X[:, 0] = label_encoder_X.fit_transform(X[:, 0]) # fitted label_encoded_X obj to the first column country. It will no longer consist of names of the counteries.

#Dummy encodings. Making the countries names into 0, 1 and 2.

from sklearn.preprocessing import OneHotEncoder 
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

# Dummy encoding for Y "Purchased" column
label_encoder_Y = LabelEncoder()
Y = label_encoder_X.fit_transform(Y) # fitted label_encoded_Y obj to the column Y "purchased". It will no longer consist of YES = 1 or NO = 0.

"""
#Fitting simple linear regression to the Training set

from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(X_train, Y_train, sample_weight = None)

#Training and Testing data

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30, random_state = 0)

#Predicting the Test set results
#Create a vector containing the prediction of the salary

Y_pred = linear_reg.predict(X_test)
Z_pred = linear_reg.predict(X_train)

#Visualization the training set results

plt.scatter(X_train, Y_train, color = 'red') #Creating a scatter plot and since the matplotlib is already declared, no need to declare it again.
plt.plot(X_train, Z_pred, color= 'blue') # to list the coordinates
plt.title("Salary vs Experience(Training set)") # Title of the plot 
plt.xlabel('Years of experience') # Name of the X axis 
plt.ylabel('Salary') # Name of the Y axis  

"""
our regressor (linear_reg) is trained on the training sets.
Therefore, whether we keep the training or testing set we will obtain the same simple linear regression line 
We will get new points on the graph if we replace train to test but it would be from same regression line.
This all happened because when we trained the linear_reg on training set (LINE 54,55), we obtained the a unique
model equation which is the simple linear equation itself.
Therefore, it doesn't matter whether we change (LINE 89) or keep it same as (LINE 71), 
it will show same regression line since both are from same unique regression

"""
#Visualization the testing set results

plt.scatter(X_test, Y_test, color = 'red') #Creating a scatter plot and since the matplotlib is already declared, no need to declare it again.
plt.plot(X_test, Y_pred, color= 'blue') # to list the coordinates
plt.title("Salary vs Experience(Training set)") # Title of the plot 
plt.xlabel('Years of experience') # Name of the X axis 
plt.ylabel('Salary') # Name of the Y axis  

"""
#Feature scaling is used to avoid the highly big to dominate or small number to bd ignored because of the big variable. It could be done either with standarisation or normalization

from sklearn.preprocessing import StandardScaler
ss_X = StandardScaler()
X_train = ss_X.fit_transform(X_train) #applying ss_X to the training set, then the data must first be fit then trained 
X_test = ss_X.transform(X_test) #applying ss_X to the testing set, then the data only need to be transformed.

"""

