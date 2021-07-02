#!/usr/bin/env python
# coding: utf-8

# # 📝 Exercise M5.02
# 
# The aim of this exercise is to find out whether a decision tree
# model is able to extrapolate.
# 
# By extrapolation, we refer to values predicted by a model outside of the
# range of feature values seen during the training.
# 
# We will first load the regression data.

# In[1]:


import pandas as pd

penguins = pd.read_csv("../datasets/penguins_regression.csv")

data_columns = ["Flipper Length (mm)"]
target_column = "Body Mass (g)"

data_train, target_train = penguins[data_columns], penguins[target_column]


# <div class="admonition note alert alert-info">
# <p class="first admonition-title" style="font-weight: bold;">Note</p>
# <p class="last">If you want a deeper overview regarding this dataset, you can refer to the
# Appendix - Datasets description section at the end of this MOOC.</p>
# </div>

# First, create two models, a linear regression model and a decision tree
# regression model, and fit them on the training data. Limit the depth at
# 3 levels for the decision tree.

# In[2]:


# Write your code here.
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

linear_model = LinearRegression()
linear_model.fit(data_train, target_train)

tree = DecisionTreeRegressor(max_depth=3)
tree.fit(data_train, target_train)


# Create a testing dataset, ranging from the minimum to the maximum of the
# flipper length of the training dataset. Get the predictions of each model
# using this test dataset.

# In[3]:


# Write your code here.
import numpy as np
data_test = pd.DataFrame(np.arange(data_train[data_columns[0]].min(),
                                  data_train[data_columns[0]].max()),
                        columns=data_columns)
linear_predictions = linear_model.predict(data_test)
tree_predictions = tree.predict(data_test)


# Create a scatter plot containing the training samples and superimpose the
# predictions of both model on the top.

# In[4]:


# Write your code here.
import matplotlib.pyplot as plt
import seaborn as sns

sns.scatterplot(data=penguins, x="Flipper Length (mm)", y="Body Mass (g)",
                color="black", alpha=0.5)
plt.plot(data_test, linear_predictions,
         label="Linear regression")
plt.plot(data_test, tree_predictions, label="Decision tree")
plt.legend()
_ = plt.title("Prediction of linear model and a decision tree")


# Now, we will check the extrapolation capabilities of each model. Create a
# dataset containing the value of your previous dataset. Besides, add values
# below and above the minimum and the maximum of the flipper length seen
# during training.

# In[5]:


# Write your code here.
offset = 30
data_test = pd.DataFrame(np.arange(data_train[data_columns[0]].min() - offset,
                                   data_train[data_columns[0]].max() + offset),
                         columns=data_columns)


# Finally, make predictions with both model on this new testing set. Repeat
# the plotting of the previous exercise.

# In[6]:


# Write your code here.
linear_predictions = linear_model.predict(data_test)
tree_predictions = tree.predict(data_test)


# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.scatterplot(data=penguins, x="Flipper Length (mm)", y="Body Mass (g)",
                color="black", alpha=0.5)
plt.plot(data_test, linear_predictions,
         label="Linear regression")
plt.plot(data_test, tree_predictions, label="Decision tree")
plt.legend()
_ = plt.title("Prediction of linear model and a decision tree")


# The linear model will extrapolate using the fitted model for flipper lengths < 175 mm and > 235 mm. In fact, we are using the model parametrization to make this predictions.
# 
# As mentioned, decision trees are non-parametric models and we observe that they cannot extrapolate. For flipper lengths below the minimum, the mass of the penguin in the training data with the shortest flipper length will always be predicted. Similarly, for flipper lengths above the maximum, the mass of the penguin in the training data with the longest flipper will always be predicted.

# In[ ]:



