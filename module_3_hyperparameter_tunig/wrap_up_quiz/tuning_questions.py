#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

penguins = pd.read_csv("../datasets/penguins.csv")

columns = ["Body Mass (g)", "Flipper Length (mm)", "Culmen Length (mm)"]
target_name = "Species"

# Remove lines with missing values for the columns of interestes
penguins_non_missing = penguins[columns + [target_name]].dropna()

data = penguins_non_missing[columns]
target = penguins_non_missing[target_name]


# In[3]:


data.head()


# In[4]:


target.head()


# In[5]:


target.value_counts()


# In[14]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
model = Pipeline(steps=[
    ("preprocessor", StandardScaler()),
    ("classifier", KNeighborsClassifier()),
])


# In[15]:


model.get_params()


# In[16]:


from sklearn.model_selection import cross_validate

for n in [5, 51]:
    model.set_params(classifier__n_neighbors=n)
    cv_results = cross_validate(model, data, target, cv=10, scoring="balanced_accuracy")
    scores = cv_results["test_score"]
    print(f"Accuracy score via cross-validation with n={n}:\n"
          f"{scores.mean():.3f} +/- {scores.std():.3f}")


# In[17]:


0.942 + 0.039


# In[18]:


0.942 - 0.039


# In[19]:


model = KNeighborsClassifier(n_neighbors=5)

cv_results = cross_validate(model, data, target, cv=10, scoring="balanced_accuracy")
scores = cv_results["test_score"]
print(f"Accuracy score via cross-validation with n={5}:\n"
      f"{scores.mean():.3f} +/- {scores.std():.3f}")


# In[23]:


from sklearn.model_selection import cross_validate

model = Pipeline(steps=[
    ("preprocessor", StandardScaler()),
    ("classifier", KNeighborsClassifier()),
])

cv_results = cross_validate(model, data, target, cv=10,
                            scoring="balanced_accuracy")
cv_results["test_score"]


# which gives values between 0.88 and 1.0 with an average close to 0.95.
# 
# It is possible to change the pipeline parameters and re-run a cross-validation with:

# In[24]:


model.set_params(classifier__n_neighbors=51)
cv_results = cross_validate(model, data, target, cv=10,
                            scoring="balanced_accuracy")
cv_results["test_score"]


# which gives slightly worse test scores but the difference is not necessarily significant: they overlap a lot.
# 
# We can disable the preprocessor by setting the preprocessor parameter to None (while resetting the number of neighbors to 5) as follows:

# In[25]:


model.set_params(preprocessor=None, classifier__n_neighbors=5)
cv_results = cross_validate(model, data, target, cv=10,
                            scoring="balanced_accuracy")
cv_results["test_score"]


# In[26]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer


all_preprocessors = [
    None,
    StandardScaler(),
    MinMaxScaler(),
    QuantileTransformer(n_quantiles=100),
    PowerTransformer(method="box-cox"),
]


# In[27]:


model = Pipeline(steps=[
    ("preprocessor", StandardScaler()),
    ("classifier", KNeighborsClassifier()),
])


# In[33]:


from sklearn.model_selection import GridSearchCV

param_grid = {
    'preprocessor': all_preprocessors,
    'classifier__n_neighbors': (5, 51, 101)}

model_grid_search = GridSearchCV(model, param_grid=param_grid,
                                 n_jobs=4, cv=2)

cv_results = cross_validate(
    model_grid_search, data, target, cv=10, return_estimator=True, scoring="balanced_accuracy")


# In[34]:


for fold_idx, estimator in enumerate(cv_results["estimator"]):
    print(f"Best parameter found on fold #{fold_idx + 1}")
    print(f"{estimator.best_params_}")


# In[35]:


scores = cv_results["test_score"]
print(f"Accuracy score by cross-validation combined with hyperparameters "
      f"search:\n{scores.mean():.3f} +/- {scores.std():.3f}")


# In[36]:


scores


# In[38]:


from sklearn.model_selection import cross_validate

model = Pipeline(steps=[
    ("preprocessor", StandardScaler()),
    ("classifier", KNeighborsClassifier()),
])

param_grid = {
    'classifier__n_neighbors': (51, 101)}

model_grid_search = GridSearchCV(model, param_grid=param_grid,
                                 n_jobs=4, cv=2)

cv_results = cross_validate(
    model_grid_search, data, target, cv=10, return_estimator=True, scoring="balanced_accuracy")

for fold_idx, estimator in enumerate(cv_results["estimator"]):
    print(f"Best parameter found on fold #{fold_idx + 1}")
    print(f"{estimator.best_params_}")


# Let's do the grid search with:

# In[40]:


from sklearn.model_selection import GridSearchCV
param_grid = {
  "preprocessor": all_preprocessors,
  "classifier__n_neighbors": [5, 51, 101],
}

grid_search = GridSearchCV(
    model,
    param_grid=param_grid,
    scoring="balanced_accuracy",
    cv=10,
).fit(data, target)
#grid_search.cv_results_


# We can sort the results and focus on the columns of interest with:

# In[41]:


results = (
    pd.DataFrame(grid_search.cv_results_)
    .sort_values(by="mean_test_score", ascending=False)
)

results = results[
    [c for c in results.columns if c.startswith("param_")]
    + ["mean_test_score", "std_test_score"]
]


# In[42]:


results


# In[ ]:




