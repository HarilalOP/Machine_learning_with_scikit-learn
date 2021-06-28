#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
ames_housing = pd.read_csv("../datasets/house_prices.csv", na_values="?")
target_name = "SalePrice"
data = ames_housing.drop(columns=target_name)
target = ames_housing[target_name]


# In[2]:


ames_housing.head()


# In[3]:


numerical_features = [
    "LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2",
    "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF",
    "GrLivArea", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces",
    "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
    "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal",
]

data_numerical = data[numerical_features]


# In[4]:


from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

linear_regression = make_pipeline(SimpleImputer(), StandardScaler(),
                                  LinearRegression())
dt_regression = make_pipeline(SimpleImputer(), DecisionTreeRegressor(random_state=0))


# In[5]:


lr_cv_results = cross_validate(linear_regression, data_numerical, target,
                            cv=10, scoring="r2",
                            return_train_score=True,
                            return_estimator=True)

dt_cv_results = cross_validate(dt_regression, data_numerical, target,
                            cv=10, scoring="r2",
                            return_train_score=True,
                            return_estimator=True)


# In[6]:


print(f"R2 of linear regresion model on the train set:\n"
      f"{lr_cv_results['train_score'].mean():.3f} +/- {lr_cv_results['train_score'].std():.3f}")

print(f"R2 of linear regresion model on the train set:\n"
      f"{lr_cv_results['test_score'].mean():.3f} +/- {lr_cv_results['test_score'].std():.3f}")


# In[7]:


print(f"R2 of DT regresion model on the train set:\n"
      f"{dt_cv_results['train_score'].mean():.3f} +/- {dt_cv_results['train_score'].std():.3f}")

print(f"R2 of linear regresion model on the train set:\n"
      f"{dt_cv_results['test_score'].mean():.3f} +/- {dt_cv_results['test_score'].std():.3f}")


# In[8]:


dt_regression.get_params()


# In[9]:


import numpy as np
from sklearn.model_selection import GridSearchCV

param_grid = {'decisiontreeregressor__max_depth': np.arange(1, 16)}
model_grid_search = GridSearchCV(dt_regression, param_grid=param_grid,
                                 n_jobs=4, cv=10)
model_grid_search.fit(data_numerical, target)


# In[10]:


search = GridSearchCV(dt_regression, param_grid=param_grid, cv=10)
cv_results_tree_optimal_depth = cross_validate(
    search, data_numerical, target, cv=10, return_estimator=True, n_jobs=2,
)
cv_results_tree_optimal_depth["test_score"].mean()


# In[11]:


import seaborn as sns
sns.set_context("talk")

max_depth = [
    estimator.best_params_["decisiontreeregressor__max_depth"]
    for estimator in cv_results_tree_optimal_depth["estimator"]
]
max_depth = pd.Series(max_depth, name="max depth")
sns.swarmplot(max_depth)


# In[12]:


from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OrdinalEncoder

categorical_processor = make_pipeline(
    SimpleImputer(strategy="constant", fill_value="missing"),
    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
)
numerical_processor = SimpleImputer()


preprocessor = make_column_transformer(
    (categorical_processor, selector(dtype_include=object)),
    (numerical_processor, selector(dtype_exclude=object))
)
tree = make_pipeline(preprocessor, DecisionTreeRegressor(random_state=0))


# In[13]:


cv_results = cross_validate(
    tree, data, target, cv=10, return_estimator=True, n_jobs=2
)
cv_results["test_score"].mean()


# In[ ]:




