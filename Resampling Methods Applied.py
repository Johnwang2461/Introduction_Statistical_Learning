import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.cross_validation import LeaveOneOut
from sklearn.metrics import mean_squared_error
import random

#Generating the data
np.random.seed( 3131 )
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
for j in range(1,3):
    MSE = 0
    LOOCV = LeaveOneOut(100)
    for train_index, test_index in LOOCV:
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        regr1 = linear_model.LinearRegression()
        regr1.fit(X_train, Y_train)
        Y_pred = regr1.coef_ + regr1.intercept_*X_test
        MSE += mean_squared_error(Y_test, Y_pred)

    print (MSE/100)