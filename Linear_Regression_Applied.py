#Standard Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from sklearn import linear_model
from sklearn.feature_selection import f_regression

#Importing Auto Data
auto = pd.read_csv('Auto.csv')
auto['horsepower'] = pd.to_numeric(auto['horsepower'], errors='coerce')
auto =  auto.dropna()

#Creating Linear Regression Object
regr = linear_model.LinearRegression()
X= np.reshape(auto['horsepower'], (len(auto['horsepower']),1))
Y= np.reshape(auto['mpg'], (len(auto['mpg']),1))

#Simple linear regression with mpg as response and horsepower as the predictor
regr.fit = regr.fit(X,Y)

# print('Intercepts: ', regr.intercept_)
# print('Coefficients: ', regr.coef_)
# # The mean square error
# print("Residual sum of squares: %.2f"
#       % np.mean((regr.predict(X) - Y) ** 2))
# # Explained variance score: 1 is perfect prediction
# print('Variance score: %.2f' % regr.score(X, Y))

# Plot outputs
# plt.scatter(X, Y,  color='black')
# plt.plot(X, regr.predict(X), color='blue',
#          linewidth=3)
#
# plt.title('Auto Dataset: Mpg vs. Horsepower')
# plt.ylabel('Miles per Gallon')
# plt.xlabel('Horsepower')
#
# plt.show()

#Scatterplot Matrix
# from pandas.tools.plotting import scatter_matrix
#
# scatter_matrix(auto)
# plt.show()
#
# print auto.corr()

#Creating Multiple Regression Object
mult_regr = linear_model.LinearRegression()
Mauto = auto.drop('mpg',1)
Mauto = Mauto.drop('name', 1)

MX = np.reshape(Mauto, (len(Mauto), 7))
MY = np.reshape(auto['mpg'], (len(auto),1))

#Multiple linear regression with mpg as the response and the other variables as the predictors
mult_regr.fit(MX,MY)

# print ('Columns: ', Mauto.columns.values.tolist())
# print('Intercepts: ', mult_regr.intercept_)
# print('Coefficients: ', mult_regr.coef_)
# # The mean square error
# print("Residual sum of squares: %.2f"
#       % np.mean((mult_regr.predict(MX) - MY) ** 2))
# # Explained variance score: 1 is perfect prediction
# print('Variance score: %.2f' % mult_regr.score(MX, MY))

#Interactions
#* Interaction
Mauto['displacementXcylinders'] = Mauto['displacement']*Mauto['cylinders']
Mauto['displacementXweight'] = Mauto['displacement']*Mauto['weight']

interaction_regr = linear_model.LinearRegression()
IMauto = Mauto[['displacement','cylinders','weight','displacementXcylinders','displacementXweight']]
IMX = np.reshape(IMauto, (len(IMauto), 5))
IMY = np.reshape(auto['mpg'], (len(auto),1))

interaction_regr.fit(IMX,IMY)

# print ('Columns: ', IMauto.columns.values.tolist())
# print('Intercepts: ', interaction_regr.intercept_)
# print('Coefficients: ', interaction_regr.coef_)
# # The mean square error
# print("Residual sum of squares: %.2f"
#       % np.mean((interaction_regr.predict(IMX) - IMY) ** 2))
# # Explained variance score: 1 is perfect prediction
# print('Variance score: %.2f' % interaction_regr.score(IMX, IMY))

#Transformations
Mauto = Mauto.drop('displacementXcylinders',1)
Mauto = Mauto.drop('displacementXweight', 1)

logMauto = np.log(Mauto)

log_regr = linear_model.LinearRegression()
logX = np.reshape(logMauto, (len(logMauto),7))
logY = np.reshape(auto['mpg'], (len(auto),1))
log_regr.fit(logX,logY)

# print ('Columns: ', logMauto.columns.values.tolist())
# print('Intercepts: ', log_regr.intercept_)
# print('Coefficients: ', log_regr.coef_)
# # The mean square error
# print("Residual sum of squares: %.2f"
#       % np.mean((log_regr.predict(logX) - logY) ** 2))
# # Explained variance score: 1 is perfect prediction
# print('Variance score: %.2f' % log_regr.score(logX, logY))

sqrtMauto = np.sqrt(Mauto)

sqrt_regr = linear_model.LinearRegression()
sqrtX = np.reshape(sqrtMauto, (len(sqrtMauto), 7))
sqrtY = np.reshape(auto['mpg'], (len(auto),1))
sqrt_regr.fit(sqrtX,sqrtY)

# print ('Columns: ', sqrtMauto.columns.values.tolist())
# print('Intercepts: ', sqrt_regr.intercept_)
# print('Coefficients: ', sqrt_regr.coef_)
# # The mean square error
# print("Residual sum of squares: %.2f"
#       % np.mean((sqrt_regr.predict(sqrtX) - sqrtY) ** 2))
# # Explained variance score: 1 is perfect prediction
# print('Variance score: %.2f' % sqrt_regr.score(sqrtX, sqrtY))


squaredMauto = np.square(Mauto)

squared_regr = linear_model.LinearRegression()
squaredX = np.reshape(squaredMauto, (len(squaredMauto), 7))
squaredY = np.reshape(auto['mpg'], (len(auto),1))
squared_regr.fit(squaredX,squaredY)

# print ('Columns: ', squaredMauto.columns.values.tolist())
# print('Intercepts: ', squared_regr.intercept_)
# print('Coefficients: ', squared_regr.coef_)
# # The mean square error
# print("Residual sum of squares: %.2f"
#       % np.mean((squared_regr.predict(squaredX) - squaredY) ** 2))
# # Explained variance score: 1 is perfect prediction
# print('Variance score: %.2f' % squared_regr.score(squaredX, squaredY))

#Load Boston Dataset
from sklearn.datasets import load_boston
boston = load_boston()
columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
boston = pd.DataFrame(data=np.column_stack([boston.data, boston.target]), columns=columns)

#Simple Linear Regressions
CRIMregr = linear_model.LinearRegression()
ZNX = np.reshape(boston['ZN'], (len(boston),1))
CRIMY = np.reshape(boston['CRIM'], (len(boston),1))

CRIMregr.fit(ZNX,CRIMY)
# print ('Column Descriptor: ', 'proportion of residential land zoned for lots over 25,000 sq.ft')
# print ('Intercepts: ', CRIMregr.intercept_)
# print ('Coefficients: ', CRIMregr.coef_)
# print ('Residual sum of squares: ', np.mean((CRIMregr.predict(ZNX)-CRIMY) ** 2))
# print ('Variance Score: ', CRIMregr.score(ZNX, CRIMY))

CRIMregr = linear_model.LinearRegression()
CHASX = np.reshape(boston['CHAS'], (len(boston),1))
CRIMY = np.reshape(boston['CRIM'], (len(boston),1))

CRIMregr.fit(CHASX,CRIMY)
print ('Column Descriptor: ', 'Charles River dummy variable (= 1 if tract bounds river; 0 otherwise) ')
print ('Intercepts: ', CRIMregr.intercept_)
print ('Coefficients: ', CRIMregr.coef_)
print ('Residual sum of squares: ', np.mean((CRIMregr.predict(CHASX)-CRIMY) ** 2))
print ('Variance Score: ', CRIMregr.score(CHASX, CRIMY))