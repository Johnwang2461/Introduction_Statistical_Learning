import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.cross_validation import LeaveOneOut
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

#Generating the data
np.random.seed( 313111 )
Y=np.random.normal(size=100)
X=np.random.normal(size=100)
Y= np.reshape(X-2+X**2+np.random.normal(size=100), (100,1))
X= np.reshape(np.random.normal(size=100), (100,1))

# plt.scatter(X, Y,  color='black')
#
# plt.title('Y vs. X')
# plt.ylabel('Y')
# plt.xlabel('X')
#
# plt.show()

#Performing LOOCV
MSE = 0
LOOCV = LeaveOneOut(100)
for train_index, test_index in LOOCV:
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    regr1 = linear_model.LinearRegression()
    regr1.fit(X_train, Y_train)
    MSE += mean_squared_error(Y_test, regr1.predict(X_test))
# print (MSE/100)

MSE2 = 0
for train_index, test_index in LOOCV:
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    regr2 = make_pipeline(PolynomialFeatures(2), linear_model.LinearRegression())
    regr2.fit(X_train, Y_train)
    MSE2 += mean_squared_error(Y_test, regr2.predict(X_test))
# print MSE2/100

MSE3 = 0
for train_index, test_index in LOOCV:
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    regr3 = make_pipeline(PolynomialFeatures(3), linear_model.LinearRegression())
    regr3.fit(X_train, Y_train)
    MSE3 += mean_squared_error(Y_test, regr3.predict(X_test))
# print MSE3/100

MSE4 = 0
for train_index, test_index in LOOCV:
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    regr4 = make_pipeline(PolynomialFeatures(4), linear_model.LinearRegression())
    regr4.fit(X_train, Y_train)
    MSE4 += mean_squared_error(Y_test, regr4.predict(X_test))
# print MSE4/100

#Boston
from sklearn.datasets import load_boston
boston = load_boston()
columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
boston = pd.DataFrame(data=np.column_stack([boston.data, boston.target]), columns=columns)

#Estimate for population mean
mean = np.mean(boston['MEDV'])
print "Estimate for Population Mean: ",mean

#Standard Error of the mean
print "Standard Error of the Mean: ",np.std(boston['MEDV'])/np.sqrt(len(boston))

#Bootstrapping for standard error
from sklearn.utils import resample
resampled_MEDV = resample(boston['MEDV'])
resampled_SE = np.std(resampled_MEDV)/np.sqrt(len(resampled_MEDV))
print "Bootstrapped Standard Error of the Mean: ",resampled_SE

#95% confidence interval
print "95% Confidence Interval: [",mean-2*resampled_SE,",",mean+2*resampled_SE,"]"

#Estimate for population median
median = np.median(boston['MEDV'])

#Bootstrapping for standard error of median
resampled_medians = []
for i in range(0,1000):
    resampled_medians.append(np.median(resample(boston['MEDV'])))
print "Average Median: ",np.mean(resampled_medians)
print "Bootstrapped Standard Error of the Median",np.std(resampled_medians)/np.sqrt(len(resampled_medians))



