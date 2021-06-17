#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

blood_transfusion = pd.read_csv("../datasets/blood_transfusion.csv")
target_name = "Class"
data = blood_transfusion.drop(columns=target_name)
target = blood_transfusion[target_name]


# In[2]:


data.head()


# In[6]:


target.unique()


# In[7]:


target.value_counts(normalize=True)


# In[8]:


from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier(strategy="most_frequent")


# In[9]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(dummy_clf, data, target, cv=10)

print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


# In[10]:


scores = cross_val_score(dummy_clf, data, target, cv=10, scoring="balanced_accuracy")

print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


# In[17]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

model = make_pipeline(StandardScaler(), KNeighborsClassifier())
model


# In[19]:


model.get_params()


# In[23]:


from sklearn.model_selection import cross_validate

cv_results = cross_validate(
    model, data, target, cv=10, scoring="balanced_accuracy", return_train_score=True
)
cv_results = pd.DataFrame(cv_results)
cv_results[["train_score", "test_score"]].mean()


# In[27]:


from sklearn.model_selection import validation_curve

model = make_pipeline(StandardScaler(), KNeighborsClassifier())

param_range = [1, 2, 5, 10, 20, 50, 100, 200, 500]
train_scores, test_scores = validation_curve(
    model, data, target, param_name="kneighborsclassifier__n_neighbors", param_range=param_range,
    cv=5, scoring="balanced_accuracy")


# In[30]:


import matplotlib.pyplot as plt

plt.plot(param_range, train_scores.mean(axis=1), label="Training score")
plt.plot(param_range, test_scores.mean(axis=1), label="Testing score")
plt.legend()

plt.xlabel("Number of neighbors of KNN")
plt.ylabel("Balanced accuracy")
_ = plt.title("Validation curve for KNN classifier")


# In[31]:


from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt

param_range = [1, 2, 5, 10, 20, 50, 100, 200, 500]
param_name = "kneighborsclassifier__n_neighbors"
train_scores, test_scores = validation_curve(
    model, data, target, param_name=param_name, param_range=param_range, cv=5,
    n_jobs=2, scoring="balanced_accuracy")

_, ax = plt.subplots()
for name, scores in zip(
    ["Training score", "Testing score"], [train_scores, test_scores]
):
    ax.plot(
        param_range, scores.mean(axis=1), linestyle="-.", label=name,
        alpha=0.8)
    ax.fill_between(
        param_range, scores.mean(axis=1) - scores.std(axis=1),
        scores.mean(axis=1) + scores.std(axis=1),
        alpha=0.5, label=f"std. dev. {name.lower()}")

ax.set_xticks(param_range)
ax.set_xscale("log")
ax.set_xlabel("Value of hyperparameter n_neighbors")
ax.set_ylabel("Balanced accuracy score")
ax.set_title("Validation curve of K-nearest neighbors")


# In[ ]:




