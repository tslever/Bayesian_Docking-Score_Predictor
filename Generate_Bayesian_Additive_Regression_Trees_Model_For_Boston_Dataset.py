from ISLP.bart import BART
from ISLP import load_data
from ISLP.models import ModelSpec as MS
import numpy as np
import pandas as pd
import sklearn.model_selection as skm

'''
We demonstrate a Python implementation of Bayesian Additive Regression Trees (BART) found in the ISLP.bart package.
We fit a model to the Boston housing data set.
This BART estimator is designed for quantitative outcome variables,
though other implementations are available for fitting logistic and probit models to categorical outcomes.
'''

bart_boston = BART(random_state = 0, burnin = 5, ndraw = 15)
Boston = load_data('Boston')
model = MS(Boston.columns.drop('medv'), intercept = False)
D = model.fit_transform(Boston)
feature_names = list(D.columns)
X = np.asarray(D)
(X_train, X_test, y_train, y_test) = skm.train_test_split(X, Boston['medv'], test_size = 0.3, random_state = 0)
bart_boston.fit(X_train, y_train)
yhat_test = bart_boston.predict(X_test.astype(np.float32))

'''
The test error of BART depends on the data set and the split into test and training data sets.
On this data set, with this split into test and training, we see that the test error of BART is similar to that of Random Forest.
'''

print(np.mean((y_test - yhat_test)**2))

'''
We can check how many times each variable appeared in the collection of trees.
This gives a summary similar to the variable importance plot for Boosting and Random Forests.
'''

var_inclusion = pd.Series(bart_boston.variable_inclusion_.mean(0), index = D.columns)
print(var_inclusion)