#!/usr/bin/env python
# coding: utf-8

# # 📝 Exercise M1.04
# 
# The goal of this exercise is to evaluate the impact of using an arbitrary
# integer encoding for categorical variables along with a linear
# classification model such as Logistic Regression.
# 
# To do so, let's try to use `OrdinalEncoder` to preprocess the categorical
# variables. This preprocessor is assembled in a pipeline with
# `LogisticRegression`. The statistical performance of the pipeline can be
# evaluated by cross-validation and then compared to the score obtained when
# using `OneHotEncoder` or to some other baseline score.
# 
# First, we load the dataset.

# In[1]:


import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census.csv")


# In[2]:


target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns=[target_name, "education-num"])


# In the previous notebook, we used `sklearn.compose.make_column_selector` to
# automatically select columns with a specific data type (also called `dtype`).
# Here, we will use this selector to get only the columns containing strings
# (column with `object` dtype) that correspond to categorical features in our
# dataset.

# In[3]:


from sklearn.compose import make_column_selector as selector

categorical_columns_selector = selector(dtype_include=object)
categorical_columns = categorical_columns_selector(data)
data_categorical = data[categorical_columns]


# We filter our dataset that it contains only categorical features.
# Define a scikit-learn pipeline composed of an `OrdinalEncoder` and a
# `LogisticRegression` classifier.
# 
# Because `OrdinalEncoder` can raise errors if it sees an unknown category at
# prediction time, you can set the `handle_unknown="use_encoded_value"` and
# `unknown_value` parameters. You can refer to the
# [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html)
# for more details regarding these parameters.

# In[5]:


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression

# Write your code here.
model = make_pipeline(
    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), LogisticRegression(max_iter=500)
)


# Your model is now defined. Evaluate it using a cross-validation using
# `sklearn.model_selection.cross_validate`.

# In[7]:


from sklearn.model_selection import cross_validate

cv_results = cross_validate(model, data_categorical, target)

scores = cv_results["test_score"]
print("The mean cross-validation accuracy is: "
      f"{scores.mean():.3f} +/- {scores.std():.3f}")


# Now, we would like to compare the statistical performance of our previous
# model with a new model where instead of using an `OrdinalEncoder`, we will
# use a `OneHotEncoder`. Repeat the model evaluation using cross-validation.
# Compare the score of both models and conclude on the impact of choosing a
# specific encoding strategy when using a linear model.

# In[8]:


from sklearn.preprocessing import OneHotEncoder

# Write your code here.
model = make_pipeline(
    OneHotEncoder(handle_unknown="ignore"), 
    LogisticRegression(max_iter=500)
)

cv_results = cross_validate(model, data_categorical, target)

scores = cv_results["test_score"]
print("The mean cross-validation accuracy is: "
      f"{scores.mean():.3f} +/- {scores.std():.3f}")


# With the linear classifier chosen, using an encoding that does not assume any ordering lead to much better result.
# 
# The important message here is: linear model and OrdinalEncoder are used together only for ordinal categorical features, features with a specific ordering. Otherwise, your model will perform poorly.

# In[ ]:




