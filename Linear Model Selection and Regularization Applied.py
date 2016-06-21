import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline

college = pd.read_csv("College.csv")
apps = np.reshape(college['Apps'], (len(college), 1))
college = college.drop(['Apps','Unnamed: 0','Private'],1)
# #First scale the data so that each feature has zero mean and unit standard deviations
# college = preprocessing.scale(college)
# apps = preprocessing.scale(apps)


# Splitting data into a training and a test set
np.random.seed(11)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(college, apps, test_size=.4)

#Least Squares Regression
lsregr = linear_model.LinearRegression()
lsregr.fit(X_train,y_train)
RSS = np.mean((lsregr.predict(X_test)-y_test) ** 2)
# print "Residual sum of squares: ", RSS

# Ridge Regression

#Alpha Value Ranges
start = .001
stop = .004
step = .001
alpha = np.arange(start,stop,step)

#Performing the Ridge Regression and acquiring Cross-validation MSEs.
rregr = linear_model.RidgeCV(alphas= alpha, normalize= True, store_cv_values=True)
rregr.fit(X_train,y_train)

# Tracking the average Cross Validation MSE values for each alpha
lowest_CV = 10000000
lowest_CV_alpha = 0
# for CV_group in range(0,int(round((stop-start)/step)), 1):
#     a = np.zeros(shape=(len(X_train),1))
#     temp = 0
#     for i in range(0,(int(len(rregr.cv_values_)*(round((stop-start)/step)))),1):
#         if i%(int(round((stop-start)/step))) == CV_group:
#             a[temp] = rregr.cv_values_.flat[i]
#             temp += 1
#     print np.mean(a)
#     if lowest_CV > np.mean(a):
#         lowest_CV = np.mean(a)
#         lowest_CV_alpha = start + step*CV_group
# print "Alpha Value Corresponding to Lowest CV: ", lowest_CV_alpha
# print "Lowest CV: ",lowest_CV

#Ridge Regression with optimized Alpha Value
rregr = linear_model.Ridge(alpha=lowest_CV_alpha, normalize=True)
rregr.fit(X_train, y_train)
RSS = np.mean((rregr.predict(X_test)-y_test)**2)
# print "Residual sum of squares: ", RSS

#Lasso Regress

#Alpha Value Ranges
start = 1
stop = 2
step = 1
alpha = np.arange(start,stop,step)

#Performing the Lasso Regression and Acquiring Cross-validation MSEs.
laregr = linear_model.LassoCV(alphas=alpha, cv=len(X_train), random_state=0)
laregr.fit(X_train,y_train)
laregr.mse_path_= laregr.mse_path_.transpose()

# Tracking the average Cross Validation MSE values for each alpha
lowest_CV = 10000000
lowest_CV_alpha = 0


for CV_group in range(0,int(round((stop-start)/step)), 1):
    a = np.zeros(shape=(len(X_train),1))
    temp = 0
    for i in range(0,(int(len(laregr.mse_path_)*(round((stop-start)/step)))),1):
        if i%(int(round((stop-start)/step))) == CV_group:
            a[temp] = laregr.mse_path_.flat[i]
            temp += 1
    if lowest_CV > np.mean(a):
        lowest_CV = np.mean(a)
        lowest_CV_alpha = start + step*CV_group
print "Alpha Value Corresponding to Lowest CV: ", lowest_CV_alpha
print "Lowest CV: ",lowest_CV

#Lasso Regression with optimized Alpha Value
laregr = linear_model.Lasso(alpha = lowest_CV_alpha, normalize=True)
laregr.fit(X_train,y_train)
RSS = np.mean((laregr.predict(X_test)-y_test)**2)
print "Residual sum of squares: ", RSS

#Principal C