#!/usr/bin/env python
# coding: utf-8

# # 📝 Exercise M1.02
# 
# The goal of this exercise is to fit a similar model as in the previous
# notebook to get familiar with manipulating scikit-learn objects and in
# particular the `.fit/.predict/.score` API.

# Let's load the adult census dataset with only numerical variables

# In[1]:


import pandas as pd
adult_census = pd.read_csv("../datasets/adult-census-numeric.csv")
data = adult_census.drop(columns="class")
target = adult_census["class"]


# In the previous notebook we used `model = KNeighborsClassifier()`. All
# scikit-learn models can be created without arguments, which means that you
# don't need to understand the details of the model to use it in scikit-learn.
# 
# One of the `KNeighborsClassifier` parameters is `n_neighbors`. It controls
# the number of neighbors we are going to use to make a prediction for a new
# data point.
# 
# What is the default value of the `n_neighbors` parameter? Hint: Look at the
# help inside your notebook `KNeighborsClassifier?` or on the [scikit-learn
# website](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

# Create a `KNeighborsClassifier` model with `n_neighbors=50`

# In[2]:


# Write your code here.
from sklearn import neighbors
model = neighbors.KNeighborsClassifier(n_neighbors=50)


# Fit this model on the data and target loaded above

# In[3]:


# Write your code here.
model.fit(data, target)


# Use your model to make predictions on the first 10 data points inside the
# data. Do they match the actual target values?

# In[4]:


# Write your code here.
target_predicted = model.predict(data[:10])
target_predicted == target[:10]


# In[8]:


number_of_correct_predictions = (
    target_predicted == target[:10]).sum()
number_of_predictions = len(target_predicted)
print(
    f"{number_of_correct_predictions}/{number_of_predictions} "
    "of predictions are correct")


# Compute the accuracy on the training data.

# In[9]:


# Write your code here.
accuracy = model.score(data, target)
print(accuracy)


# Now load the test data from `"../datasets/adult-census-numeric-test.csv"` and
# compute the accuracy on the test data.

# In[10]:


# Write your code here.
test_data = pd.read_csv("../datasets/adult-census-numeric-test.csv")
data_test = test_data.drop(columns="class")
target_test = test_data["class"]
model.score(data_test, target_test)


# In[ ]:




