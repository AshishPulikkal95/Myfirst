#Importing the libraries
import numpy as np #Contains mathematical tools and used to include any type of mathematical codes
import matplotlib.pyplot as plt #This allows to plot charts 
import pandas as pd #Import and manage datasets


#Import the datasets using pandas 
dataset = pd.read_csv("Data.csv")

#Matrix of features. In this X is the independent data and Y is the dependent data.
X = dataset.iloc[:, :-1].values #This line means that we are taking all the columns except the last ome 
Y = dataset.iloc[:,3].values #This line means that we are taking the last column of that is "Purchased" 

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


#Training and Testing data

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)



#Feature scaling is used to avoid the highly huge or small number to avoid domination of one variable over another. It could be done either with standarisation or normalization

from sklearn.preprocessing import StandardScaler
ss_X = StandardScaler()
X_train = ss_X.fit_transform(X_train) #applying ss_X to the training set, then the data must first be fit then trained 
X_test = ss_X.transform(X_test) #applying ss_X to the testing set, then the data only need to be transformed.