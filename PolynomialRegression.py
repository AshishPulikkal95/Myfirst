# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 14:56:05 2018

@author: Ashish
"""

# Polynomial Regression: This is not a linear regressors.

"""
Does not make much sense if we have small dataset and we split it into training and testing dataset. Therefore, never split if the dataset is small in size.
No feature scaling is needed because polynomial regressor consist of adding some polynomial terms into the multiple linear regression equation and therefore we will use the same linear regression library used to build simple and multiple linear regression models.
"""
# ----------Data Preprocessing------------- 

# Importing the libraries

import numpy as np
import pandas as pd

# Importing the dataset

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values



"""
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)

"""
"""
Linear regression is to compare the results of it with that of Polynomial model
"""

#----------Fitting Linear Regression to the dataset----------------
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(X, Y)

#----------Fitting Polynomial regression to the dataset------------

from sklearn.preprocessing import PolynomialFeatures
pol_reg = PolynomialFeatures(degree = 2)
X_poly = pol_reg.fit_transform(X)
lin_reg_2 = LinearRegression ()
lin_reg_2.fit(X_poly, Y)


#---------------Visualization of the linear and Polynomial results---------------------

#..........Linear-------------

import matplotlib.pyplot as plt
plt.scatter(X, Y, color = 'red')
plt.plot(X, linear_reg.predict(X), color = 'blue')
plt.title('Linear Model results')
plt.xlabel('Independent Variable')
plt.ylabel('Dependent Variable')

#--------Polynomial-----------
x_grid = np.arange(min(X), max(X), 0.1) # x_grid contains all the levels plus incremented steps between the levels with a resolution of 0.1
x_grid = x_grid.reshape((len(x_grid), 1)) # Since x_grid is a vector and we need matrix 
plt.scatter(X, Y, color ='red')
plt.plot(x_grid, lin_reg_2.predict(pol_reg.fit_transform(x_grid)), color = 'blue' )
plt.title('Polynomial Model results')
plt.xlabel('Independent variables')
plt.ylabel('Dependent variables')

#----------Predict new results with Linear and Polynomial Regression-----------

#------Linear--------
linear_reg.predict(X) # Specify the level in place of X to get a specific prediction
linear_reg.predict(6.5)

#------Polynomial------
lin_reg_2.predict(pol_reg.fit_transform(6.5))