import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn import ensemble
from sklearn.metrics import mean_squared_error

np.random.seed(1337)

#Importing the data
Hitters = pd.read_csv("Hitters.csv")

#Cleaning the data by getting rid of NaNs and log transforming salary
# print Hitters.isnull().sum() #59 NaN in Salary
Hitters = Hitters.dropna()
Hitters['Salary'] = np.log(Hitters['Salary'])

#Create the Response for our supervised learning technique
Salary = Hitters['Salary']
Hitters = Hitters.drop(['Unnamed: 0','Salary'], axis=1)

#To avoid any issues, we convert every int column into a float and every categorical variable into dummy variables
for column in list(Hitters.columns.values):
    if Hitters[column].dtypes == 'int64':
        Hitters[column] = Hitters[column].astype(float)

Hitters = pd.get_dummies(Hitters)

#Creation of training and test sets
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(Hitters,Salary, test_size=.2375)

#Tune the learning_rate parameter to determine which will provide the lowest mean squared error
training_mse = {}
test_mse = {}
param_search = np.arange(0.0001, .0101, .0001)
for i in param_search:
    Gradient_Boosting_Regerssion = ensemble.GradientBoostingRegressor(n_estimators=1000, learning_rate=i)
    Gradient_Boosting_Regerssion.fit(X_train, Y_train)
    training_mse[i] = mean_squared_error(Y_train, Gradient_Boosting_Regerssion.predict(X_train))
    test_mse[i] = mean_squared_error(Y_test, Gradient_Boosting_Regerssion.predict(X_test))
    # mse.append(mean_squared_error(Y_test, Gradient_Boosting_Regerssion.predict(X_test)))

# print "Lowest Training MSE key value pair: ", min(training_mse, key=training_mse.get), " ", min(training_mse.values())
# print "Lowest Test MSE key value pair: ", min(test_mse, key=test_mse.get)," ", min(test_mse.values())

#Displaying the Mean Squared Errors
# fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))
# ax1.plot(param_search, training_mse.values(), '-v')
# ax2.plot(param_search, test_mse.values(), '-v')
# ax1.set_title('GBR: Training MSE')
# ax2.set_title('GBR: Test MSE')
# ax1.set_ylabel('Training MSE')
# ax2.set_xlabel('Test MSE')
#
# for ax in fig.axes:
#     ax.set_xlabel('Shrinkage Parameter Value')

# plt.show()

#Displaying the Relative Importance of Features
# print "Feature Distribution: ", Gradient_Boosting_Regerssion.feature_importances_
# ticks = np.arange(len(list(Hitters.columns.values)))
# plt.bar(ticks, Gradient_Boosting_Regerssion.feature_importances_, align='center', alpha=0.5, width=.33)
# plt.xticks((ticks), list(Hitters.columns.values))
# plt.ylabel('Relative Importance of Features')
# plt.title('Importance of Each Feature')
# plt.show()

#Bagging
BaggingRegressor = ensemble.BaggingRegressor(n_estimators=1000)
BaggingRegressor.fit(X_train, Y_train)
Training_MSE = mean_squared_error(Y_train, BaggingRegressor.predict(X_train))
Test_MSE = mean_squared_error(Y_test, BaggingRegressor.predict(X_test))

print "Bagging Decision Tree Training MSE: ", Training_MSE
print "Bagging Decision Tree Test MSE: ", Test_MSE