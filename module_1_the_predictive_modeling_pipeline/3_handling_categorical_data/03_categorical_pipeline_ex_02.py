#!/usr/bin/env python
# coding: utf-8

# # 📝 Exercise M1.05
# 
# The goal of this exercise is to evaluate the impact of feature preprocessing
# on a pipeline that uses a decision-tree-based classifier instead of logistic
# regression.
# 
# - The first question is to empirically evaluate whether scaling numerical
#   feature is helpful or not;
# - The second question is to evaluate whether it is empirically better (both
#   from a computational and a statistical perspective) to use integer coded or
#   one-hot encoded categories.

# In[1]:


import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census.csv")


# In[2]:


target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns=[target_name, "education-num"])


# As in the previous notebooks, we use the utility `make_column_selector`
# to only select column with a specific data type. Besides, we list in
# advance all categories for the categorical columns.

# In[3]:


from sklearn.compose import make_column_selector as selector

numerical_columns_selector = selector(dtype_exclude=object)
categorical_columns_selector = selector(dtype_include=object)
numerical_columns = numerical_columns_selector(data)
categorical_columns = categorical_columns_selector(data)


# ## Reference pipeline (no numerical scaling and integer-coded categories)
# 
# First let's time the pipeline we used in the main notebook to serve as a
# reference:

# In[4]:


get_ipython().run_cell_magic('time', '', 'from sklearn.model_selection import cross_validate\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.compose import ColumnTransformer\nfrom sklearn.preprocessing import OrdinalEncoder\nfrom sklearn.experimental import enable_hist_gradient_boosting\nfrom sklearn.ensemble import HistGradientBoostingClassifier\n\ncategorical_preprocessor = OrdinalEncoder(handle_unknown="use_encoded_value",\n                                          unknown_value=-1)\npreprocessor = ColumnTransformer([\n    (\'categorical\', categorical_preprocessor, categorical_columns)],\n    remainder="passthrough")\n\nmodel = make_pipeline(preprocessor, HistGradientBoostingClassifier())\ncv_results = cross_validate(model, data, target)\nscores = cv_results["test_score"]\nprint("The mean cross-validation accuracy is: "\n      f"{scores.mean():.3f} +/- {scores.std():.3f}")')


# ## Scaling numerical features
# 
# Let's write a similar pipeline that also scales the numerical features using
# `StandardScaler` (or similar):

# In[7]:


get_ipython().run_cell_magic('time', '', '# Write your code here.\n\nfrom sklearn.preprocessing import StandardScaler\n\ncategorical_preprocessor = OrdinalEncoder(handle_unknown="use_encoded_value",\n                                          unknown_value=-1)\nnumerical_preprocessor = StandardScaler()\n\npreprocessor = ColumnTransformer([\n    (\'categorical\', categorical_preprocessor, categorical_columns),\n    (\'standard-scaler\', numerical_preprocessor, numerical_columns)])\n\nmodel = make_pipeline(preprocessor, HistGradientBoostingClassifier())\ncv_results = cross_validate(model, data, target)\nscores = cv_results["test_score"]\nprint("The mean cross-validation accuracy is: "\n      f"{scores.mean():.3f} +/- {scores.std():.3f}")')


# ### Analysis
# 
# We can observe that both the accuracy and the training time are approximately the same as the reference pipeline (any time difference you might observe is not significant).
# 
# Scaling numerical features is indeed useless for most decision tree models in general and for HistGradientBoostingClassifier in particular.

# ## One-hot encoding of categorical variables
# 
# For linear models, we have observed that integer coding of categorical
# variables can be very detrimental. However for
# `HistGradientBoostingClassifier` models, it does not seem to be the case as
# the cross-validation of the reference pipeline with `OrdinalEncoder` is good.
# 
# Let's see if we can get an even better accuracy with `OneHotEncoder`.
# 
# Hint: `HistGradientBoostingClassifier` does not yet support sparse input
# data. You might want to use
# `OneHotEncoder(handle_unknown="ignore", sparse=False)` to force the use of a
# dense representation as a workaround.

# In[9]:


get_ipython().run_cell_magic('time', '', '# Write your code here.\nfrom sklearn.preprocessing import OneHotEncoder\n\ncategorical_preprocessor = OneHotEncoder(handle_unknown="ignore", sparse=False)\n\npreprocessor = ColumnTransformer([\n    (\'one-hot-encoder\', categorical_preprocessor, categorical_columns)],\n    remainder="passthrough")\n\nmodel = make_pipeline(preprocessor, HistGradientBoostingClassifier())\ncv_results = cross_validate(model, data, target)\nscores = cv_results["test_score"]\nprint("The mean cross-validation accuracy is: "\n      f"{scores.mean():.3f} +/- {scores.std():.3f}")')


# ### Analysis
# 
# From an accuracy point of view, the result is almost exactly the same. The reason is that HistGradientBoostingClassifier is expressive and robust enough to deal with misleading ordering of integer coded categories (which was not the case for linear models).
# 
# However from a computation point of view, the training time is significantly longer: this is caused by the fact that OneHotEncoder generates approximately 10 times more features than OrdinalEncoder.
# 
# Note that the current implementation HistGradientBoostingClassifier is still incomplete, and once sparse representation are handled correctly, training time might improve with such kinds of encodings.
# 
# The main take away message is that arbitrary integer coding of categories is perfectly fine for HistGradientBoostingClassifier and yields fast training times.

# In[ ]:




