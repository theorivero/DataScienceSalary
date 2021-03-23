# -*- coding: utf-8 -*-

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('eda_data.csv')

# choose relevant columns
df.columns

df_model= df[['avg_salary', "Rating", "Size", "Type of ownership", "hourly","employer_provided",
            'same_state', 'age', 
            'python_yn', 'spark_yn', 'aws_yn', 'excel_yn', 'tensor_yn',
            'sql_yn', 'job_simp', 'seniority', 'desc_len', 'num_comp',
            'Industry', 'Sector', 'Revenue', 'job_state']]
# get dummy data
df_dum = pd.get_dummies(df_model)

# train test split
from sklearn.model_selection import train_test_split
y = df_dum['avg_salary']
X = df_dum.drop('avg_salary', axis=1)

X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size = 0.33, random_state=42)

# multiple linear regression
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


X_sm = X = sm.add_constant(X)
model = sm.OLS(y, X_sm)
model.fit().summary()

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score

lm = LinearRegression()
lm.fit(X_train, y_train)

np.mean(cross_val_score(lm, X_train, y_train, scoring = 'neg_mean_absolute_error', cv=3))

# lasso regression

lm_l = Lasso()
np.mean(cross_val_score(lm_l, X_train, y_train, scoring = 'neg_mean_absolute_error', cv=3))
alpha = []
error = []
for i in range(1,100):
    alpha.append(i/100)
    lm_l = Lasso(alpha=(i/100))
    error.append(np.mean(cross_val_score(lm_l, X_train, y_train, scoring = 'neg_mean_absolute_error', cv=3)))

plt.plot(alpha, error)

err = tuple(zip(alpha, error))
df_err = pd.DataFrame(err, columns = ['alpha', 'error'])
df_err[df_err.error == max(df_err.error)]

# random forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
np.mean(cross_val_score(rf, X_train, y_train, scoring = 'neg_mean_absolute_error', cv=3))



# tune models GridSearchCV

from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':range(10,300,10), 'criterion':('mse', 'mae'), 'max_features':('auto', 'sqrt', 'log2')}
gs = GridSearchCV(rf, parameters, scoring='neg_mean_absolute_error', cv=3)
 
gs.fit(X_train, y_train)

gs.best_score_

gs.best_estimator_

# Test emsembles

tpred_lm = lm.predict(X_test)
tpred_lml = lm_l.predict(X_test)
tpred_rf = gs.best_estimator_.predict(X_test)

from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_test, tpred_lm)
mean_absolute_error(y_test, tpred_lml)
mean_absolute_error(y_test, tpred_rf)

mean_absolute_error(y_test, (tpred_lm+tpred_rf)/2)
