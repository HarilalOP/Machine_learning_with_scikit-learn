#!/usr/bin/env python
# coding: utf-8

# # 📝 Exercise M6.03
# 
# This exercise aims at verifying if AdaBoost can over-fit.
# We will make a grid-search and check the scores by varying the
# number of estimators.
# 
# We will first load the California housing dataset and split it into a
# training and a testing set.

# In[1]:


from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

data, target = fetch_california_housing(return_X_y=True, as_frame=True)
target *= 100  # rescale the target in k$
data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=0, test_size=0.5)


# <div class="admonition note alert alert-info">
# <p class="first admonition-title" style="font-weight: bold;">Note</p>
# <p class="last">If you want a deeper overview regarding this dataset, you can refer to the
# Appendix - Datasets description section at the end of this MOOC.</p>
# </div>

# Then, create an `AbaBoostRegressor`. Use the function
# `sklearn.model_selection.validation_curve` to get training and test scores
# by varying the number of estimators. Use the mean absolute error as a metric
# by passing `scoring="neg_mean_absolute_error"`.
# *Hint: vary the number of estimators between 1 and 60.*

# In[2]:


# Write your code here.
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import validation_curve

adaboost = AdaBoostRegressor()
param_range = np.unique(np.logspace(0, 1.8, num=30).astype(int))
train_scores, test_scores = validation_curve(
    adaboost, data_train, target_train,
    param_name="n_estimators", param_range=param_range,
    scoring="neg_mean_absolute_error", n_jobs=2)
train_errors, test_errors = -train_scores, -test_scores


# Plot both the mean training and test errors. You can also plot the
# standard deviation of the errors.
# *Hint: you can use `plt.errorbar`.*

# In[3]:


# Write your code here.
import matplotlib.pyplot as plt

plt.errorbar(param_range, train_errors.mean(axis=1),
             yerr=train_errors.std(axis=1), label="Training score",
             alpha=0.7)
plt.errorbar(param_range, test_errors.mean(axis=1),
             yerr=test_errors.std(axis=1), label="Cross-validation score",
             alpha=0.7)

plt.legend()
plt.ylabel("Mean absolute error in k$\n(smaller is better)")
plt.xlabel("# estimators")
_ = plt.title("Validation curve for AdaBoost regressor")


# Plotting the validation curve, we can see that AdaBoost is not immune against
# overfitting. Indeed, there is an optimal number of estimators to be found.
# Adding too many estimators is detrimental for the statistical performance of
# the model.

# Repeat the experiment using a random forest instead of an AdaBoost regressor.

# In[4]:


# Write your code here.
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor()
train_scores, test_scores = validation_curve(
    forest, data_train, target_train,
    param_name="n_estimators", param_range=param_range,
    scoring="neg_mean_absolute_error", n_jobs=2)
train_errors, test_errors = -train_scores, -test_scores


# In[5]:


plt.errorbar(param_range, train_errors.mean(axis=1),
             yerr=train_errors.std(axis=1), label="Training score",
             alpha=0.7)
plt.errorbar(param_range, test_errors.mean(axis=1),
             yerr=test_errors.std(axis=1), label="Cross-validation score",
             alpha=0.7)

plt.legend()
plt.ylabel("Mean absolute error in k$\n(smaller is better)")
plt.xlabel("# estimators")
_ = plt.title("Validation curve for RandomForest regressor")


# In contrary to the AdaBoost regressor, we can see that increasing the number trees in the forest will increase the statistical performance (by decreasing the mean absolute error) of the random forest. In fact, a random forest has less chance to suffer from overfitting than AdaBoost when increasing the number of estimators.

# In[ ]:



