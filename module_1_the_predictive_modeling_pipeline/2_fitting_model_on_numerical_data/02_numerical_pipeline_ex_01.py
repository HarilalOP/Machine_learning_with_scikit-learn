#!/usr/bin/env python
# coding: utf-8

# # 📝 Exercise M1.03
# 
# The goal of this exercise is to compare the statistical performance of our
# classifier (81% accuracy) to some baseline classifiers that would ignore the
# input data and instead make constant predictions.
# 
# - What would be the score of a model that always predicts `' >50K'`?
# - What would be the score of a model that always predicts `' <=50K'`?
# - Is 81% or 82% accuracy a good score for this problem?
# 
# Use a `DummyClassifier` and do a train-test split to evaluate
# its accuracy on the test set. This
# [link](https://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators)
# shows a few examples of how to evaluate the statistical performance of these
# baseline models.

# In[1]:


import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census.csv")


# We will first split our dataset to have the target separated from the data
# used to train our predictive model.

# In[2]:


target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns=target_name)


# We start by selecting only the numerical columns as seen in the previous
# notebook.

# In[3]:


numerical_columns = [
    "age", "capital-gain", "capital-loss", "hours-per-week"]

data_numeric = data[numerical_columns]


# Split the dataset into a train and test sets.

# In[4]:


from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(data_numeric, target, random_state=42)


# Use a `DummyClassifier` such that the resulting classifier will always
# predict the class `' >50K'`. What is the accuracy score on the test set?
# Repeat the experiment by always predicting the class `' <=50K'`.
# 
# Hint: you can refer to the parameter `strategy` of the `DummyClassifier`
# to achieve the desired behaviour.

# In[7]:


from sklearn.dummy import DummyClassifier

class_to_predict = " >50K"
model = DummyClassifier(strategy="constant", constant=class_to_predict)
model.fit(data_train, target_train)
score = model.score(data_test, target_test)
print(f"Accuracy of a model predicting only >50K revenue: {score:.3f}")


# In[8]:


from sklearn.dummy import DummyClassifier

class_to_predict = " <=50K"
model = DummyClassifier(strategy="constant", constant=class_to_predict)
model.fit(data_train, target_train)
score = model.score(data_test, target_test)
print(f"Accuracy of a model predicting only <=50K revenue: {score:.3f}")


# We observe that this model has an accuracy higher than 0.5. This is due to the fact that we have 3/4 of the target belonging to low-revenue class.
# 
# Therefore, any predictive model giving results below this dummy classifier will not be helpful.

# In[9]:


adult_census["class"].value_counts()


# In[10]:


(target == " <=50K").mean()


# In practice, we could have the strategy "most_frequent" to predict the class that appears the most in the training target.

# In[11]:


from sklearn.dummy import DummyClassifier

model = DummyClassifier(strategy="most_frequent")
model.fit(data_train, target_train)
score = model.score(data_test, target_test)
print(f"Accuracy of a model predicting the most frequent class: {score:.3f}")


# So the LogisticRegression accuracy (roughly 81%) seems better than the DummyClassifier accuracy (roughly 76%). In a way it is a bit reassuring, using a machine learning model gives you a better performance than always predicting the majority class, i.e. the low income class " <=50K".
