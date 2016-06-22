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
MSE = np.mean((lsregr.predict(X_test)-y_test) ** 2)
# print "Mean Squared Error: ", RSS

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
MSE = np.mean((rregr.predict(X_test)-y_test)**2)
# print "Mean Squared Error: ", RSS

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


# for CV_group in range(0,int(round((stop-start)/step)), 1):
#     a = np.zeros(shape=(len(X_train),1))
#     temp = 0
#     for i in range(0,(int(len(laregr.mse_path_)*(round((stop-start)/step)))),1):
#         if i%(int(round((stop-start)/step))) == CV_group:
#             a[temp] = laregr.mse_path_.flat[i]
#             temp += 1
#     if lowest_CV > np.mean(a):
#         lowest_CV = np.mean(a)
#         lowest_CV_alpha = start + step*CV_group
# print "Alpha Value Corresponding to Lowest CV: ", lowest_CV_alpha
# print "Lowest CV: ",lowest_CV
#
# #Lasso Regression with optimized Alpha Value
# laregr = linear_model.Lasso(alpha = lowest_CV_alpha, normalize=True)
# laregr.fit(X_train,y_train)
# MSE = np.mean((laregr.predict(X_test)-y_test)**2)
# print "Mean Squared Error: ", RSS

#Principal Components Regression
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

X_train_scaled = scale(X_train)
pca = PCA(n_components='mle')
pca.fit(X_train_scaled,y_train)

# print "Principal axes in feature space, representing the directions of the maximum variance in the data: \n", pca.components_
# print "Percentage of variance explained by each of the selected components. If n_components is not set, then all components are stored and the sum of explained variance is equal to 1.0: \n",pca.explained_variance_ratio_
# print "Per-feature empirical mean, estimated from the training set: \n",pca.mean_
print "The estimated number of components. Relevant when n_components is set to 'mle' or a number between 0 and 1 to select using explained variance: \n",pca.n_components
# print "The estimated noise covariance following the Probabilistic PCA model: \n",pca.noise_variance_

print "Variance (% cumulative) explained by principal components: \n", np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100), "\n"

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

print "Mean Squared Error CV: ", mse
# fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))
# ax1.plot(mse, '-v')
# ax2.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], mse[1:17], '-v')
# ax2.set_title('Intercept excluded from plot')
#
# for ax in fig.axes:
#     ax.set_xlabel('Number of principal components in regression')
#     ax.set_ylabel('MSE')
#     ax.set_xlim((-0.2,17.2))

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

print "Mean Squared Error: ",mse[7]

plt.plot([1,2,3,4,5,6,7,8], mse[0:8], '-v')
plt.title('PCA: MSE vs. Principal Components')
plt.xlabel('Number of principal components in regression')
plt.ylabel('MSE')
plt.xlim((-0.2,8.2))
plt.show()