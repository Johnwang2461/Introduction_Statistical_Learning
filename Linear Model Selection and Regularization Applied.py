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
college = college.drop(['Apps','Unnamed: 0'],1)

#Create dummy variables for private: 0 = Yes, 1 = No
college['Private'] = pd.get_dummies(college['Private'])

# Splitting data into a training and a test set
np.random.seed(11)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(college, apps, test_size=.4)

#Least Squares Regression
lsregr = linear_model.LinearRegression()
lsregr.fit(X_train,y_train)
MSE_LS = np.mean((lsregr.predict(X_test)-y_test) ** 2)
# print "Mean Squared Error: ", MSE_LS

# Ridge Regression

#Alpha Value Ranges
start = 0.001
stop = .004
step = .001
alpha = np.arange(start,stop,step)

#Performing the Ridge Regression and acquiring Cross-validation MSEs.
rregr = linear_model.RidgeCV(alphas= alpha, normalize= True, store_cv_values=True)
rregr.fit(X_train,y_train)

# Tracking the average Cross Validation MSE values for each alpha
lowest_CV = 10000000
lowest_CV_alpha = 0
for CV_group in range(0,int(round((stop-start)/step)), 1):
    a = np.zeros(shape=(len(X_train),1))
    temp = 0
    for i in range(0,(int(len(rregr.cv_values_)*(round((stop-start)/step)))),1):
        if i%(int(round((stop-start)/step))) == CV_group:
            a[temp] = rregr.cv_values_.flat[i]
            temp += 1
    if lowest_CV > np.mean(a):
        lowest_CV = np.mean(a)
        lowest_CV_alpha = start + step*CV_group
# print "Alpha Value Corresponding to Lowest CV: ", lowest_CV_alpha
# print "Lowest CV: ",lowest_CV

#Ridge Regression with optimized Alpha Value
rregr = linear_model.Ridge(alpha=lowest_CV_alpha, normalize=True)
rregr.fit(X_train, y_train)
MSE_R = np.mean((rregr.predict(X_test)-y_test)**2)
# print "Mean Squared Error: ", MSE_R

#Lasso Regress
from sklearn.preprocessing import scale

X_train_scaled = scale(X_train)
X_test_scaled = scale(X_test)
y_train_scaled = scale(y_train)
y_test_scaled = scale(y_test)

laregr = linear_model.LassoCV(eps=10**-12, cv=len(X_train_scaled),selection='random', random_state=0, fit_intercept=False)
laregr.fit(X_train_scaled,y_train)
lowest_CV_alpha = laregr.alpha_

laregr = linear_model.Lasso(alpha = 100.2, random_state=0, fit_intercept=False)
laregr.fit(X_train_scaled,y_train)
MSE_LA = np.mean((laregr.predict(X_test_scaled)-y_test)**2)

print "Alpha Value: ", lowest_CV_alpha
print "Coefficient Values: ", laregr.coef_
print "Number of Coefficients: 13"
print "Mean Squared Error: ", MSE_LA

#Principal Components Regression
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

X_train_scaled = scale(X_train)
pca = PCA(n_components='mle')
pca.fit(X_train_scaled,y_train)

# print "Variance (% cumulative) explained by principal components: \n", np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100), "\n"

#Cross validation for PCR:
n = len(X_train_scaled)
kf_10 = cross_validation.KFold(n, n_folds=10, shuffle=True, random_state=0)

pcrregr = linear_model.LinearRegression()
mse = []

score = -1*cross_validation.cross_val_score(pcrregr, np.ones((n,1)), y_train.ravel(), cv=kf_10, scoring='mean_squared_error').mean()
mse.append(score)

for i in np.arange(1,17):
    score = -1*cross_validation.cross_val_score(pcrregr, X_train_scaled[:,:i], y_train.ravel(), cv=kf_10, scoring='mean_squared_error').mean()
    mse.append(score)

# print "Mean Squared Error CV: ", mse
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))
ax1.plot(mse, '-v')
ax2.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], mse[1:17], '-v')
ax2.set_title('Intercept excluded from plot')

for ax in fig.axes:
    ax.set_xlabel('Number of principal components in regression')
    ax.set_ylabel('MSE')
    ax.set_xlim((-0.2,17.2))

# plt.show()

#Based off of the plot, 8 principal components seem to minimize test error.
X_test_scaled = scale(X_test)
pca_test = PCA(n_components=8)
pca_test.fit(X_test_scaled,y_test)
X_test_scaled = pca_test.fit_transform(X_test_scaled)

mse = []

for i in np.arange(1,9):
    score = -1*cross_validation.cross_val_score(pcrregr, X_test_scaled[:,:i], y_test.ravel(), cv=len(X_test_scaled), scoring='mean_squared_error').mean()
    mse.append(score)
MSE_PCA = mse[7]
# print "Mean Squared Error: ",MSE_PCA
#
# plt.plot([1,2,3,4,5,6,7,8], mse[0:8], '-v')
# plt.title('PCA: MSE vs. Principal Components')
# plt.xlabel('Number of principal components in regression')
# plt.ylabel('MSE')
# plt.xlim((-0.2,8.2))
# plt.show()

#Partial Least Squares Regression
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import scale

X_train_scaled = scale(X_train)
X_test_scaled = scale(X_test)

#Performing Cross_Validation for PLS
mse = []
n=  len(X_train_scaled)
kf_10 = cross_validation.KFold(n,n_folds=10, shuffle=True, random_state=0)

for i in np.arange(1,17):
    plsregr = PLSRegression(n_components=i, scale=False)
    plsregr.fit(X_train_scaled,y_train)
    score = -1*cross_validation.cross_val_score(plsregr, X_train_scaled, y_train, cv=kf_10, scoring='mean_squared_error').mean()
    mse.append(score)

plt.plot(np.arange(1,17), np.array(mse), '-v')
plt.title("PLS: MSE vs. Principal Components")
plt.xlabel('Number of principal components in PLS regression')
plt.ylabel('MSE')
plt.xlim((-0.2, 17.2))

#Based off of the plot, 12 principal components minimized MSE
plsregr_test = PLSRegression(n_components=12, scale=False)
plsregr_test.fit(X_train_scaled, y_train)
MSE_PLS = np.mean((plsregr_test.predict(X_test_scaled) - y_test) ** 2)
# print "Mean Squared Error: ", MSE_PLS

#Compare the results from above. We use (R)^2 for all models
Test_avg= np.mean(y_test)

LS_R2 = 1 - MSE_LS/(np.mean((Test_avg-y_test)**2))
R_R2 = 1 - MSE_R/(np.mean((Test_avg-y_test)**2))
LA_R2 = 1 - MSE_LA/(np.mean((Test_avg-y_test)**2))
PCA_R2 = 1 - MSE_PCA/(np.mean((Test_avg-y_test)**2))
PLS_R2 = 1 - MSE_PLS/(np.mean((Test_avg-y_test)**2))

print "Least Squares Regression (R)^2: ", LS_R2
print "Ridge Regression (R)^2: ", R_R2
print "Lasso Regression (R)^2: ", LA_R2
print "Principal Component Analysis Regression (R)^2: ", PCA_R2
print "Partial Least Squares Regression (R)^2: ", PLS_R2