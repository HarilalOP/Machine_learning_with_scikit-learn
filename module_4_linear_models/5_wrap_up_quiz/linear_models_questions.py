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


from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate

model = make_pipeline(StandardScaler(), SimpleImputer(), LinearRegression())
cv_results = cross_validate(
    model, data_numerical, target, cv=10, return_estimator=True
)
coefs = [estimator[-1].coef_ for estimator in cv_results["estimator"]]
coefs = pd.DataFrame(coefs, columns=numerical_features)
coefs.describe().loc[["min", "max"]]


# In[5]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("talk")
# Define the style of the box style
boxplot_property = {
    "vert": True,
    "whis": 100,
    "patch_artist": True,
    "widths": 0.5,
    "rot": 90,
    "boxprops": dict(linewidth=3, color="black", alpha=0.9),
    "medianprops": dict(linewidth=2.5, color="black", alpha=0.9),
    "whiskerprops": dict(linewidth=3, color="black", alpha=0.9),
    "capprops": dict(linewidth=3, color="black", alpha=0.9),
}

_, ax = plt.subplots(figsize=(15, 10))
_ = coefs.plot.box(**boxplot_property, ax=ax)


# In[6]:


from sklearn.linear_model import Ridge

model = make_pipeline(StandardScaler(), SimpleImputer(), Ridge())
cv_results = cross_validate(
    model, data_numerical, target, cv=10, return_estimator=True
)

coefs = [estimator[-1].coef_ for estimator in cv_results["estimator"]]
coefs = pd.DataFrame(coefs, columns=numerical_features)
coefs.describe().loc[["min", "max"]]


# In[7]:


_, ax = plt.subplots(figsize=(15, 10))
_ = coefs.abs().plot.box(**boxplot_property, ax=ax)


# Indeed, we should look at the variability of the "GarageCars" coefficient during the experiment. In the previous plot, we could see that the coefficients related to this feature were varying from one fold to another. We can check the standard deviation of the coefficients and check the evolution.

# In[8]:


coefs.describe()["GarageCars"]


# In[9]:


column_to_drop = "GarageArea"
data_numerical = data_numerical.drop(columns=column_to_drop)

cv_results = cross_validate(
    model, data_numerical, target, cv=10, return_estimator=True
)
coefs = [estimator[-1].coef_ for estimator in cv_results["estimator"]]
coefs = pd.DataFrame(coefs, columns=data_numerical.columns)
coefs.describe()["GarageCars"]


# In[10]:


_, ax = plt.subplots(figsize=(15, 10))
_ = coefs.abs().plot.box(**boxplot_property, ax=ax)


# In[11]:


import numpy as np
from sklearn.linear_model import RidgeCV

alphas = np.logspace(-1, 3, num=30)
model = make_pipeline(
    StandardScaler(), SimpleImputer(), RidgeCV(alphas=alphas)
)
cv_results = cross_validate(
    model, data_numerical, target, cv=10, return_estimator=True
)
coefs = [estimator[-1].coef_ for estimator in cv_results["estimator"]]
coefs = pd.DataFrame(coefs, columns=data_numerical.columns)
_, ax = plt.subplots(figsize=(15, 10))
_ = coefs.abs().plot.box(**boxplot_property, ax=ax)


# In[12]:


alpha = [estimator[-1].alpha_ for estimator in cv_results["estimator"]]
alpha = pd.Series(alpha, name="alpha")
ax = sns.swarmplot(alpha)


# In[13]:


adult_census = pd.read_csv("../datasets/adult-census.csv")
target = adult_census["class"]
data = adult_census.select_dtypes(["integer", "floating"])
data = data.drop(columns=["education-num"])


# In[14]:


from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OrdinalEncoder

data[selector(dtype_exclude=object)]


# In[25]:


data.isnull().sum()


# In[16]:


data.info()


# In[27]:


list(data.columns)


# In[17]:


from sklearn.linear_model import LogisticRegression

model = make_pipeline(StandardScaler(), LogisticRegression())
cv_results = cross_validate(
    model, data, target, cv=10, return_estimator=True
)


# In[19]:


print(f"Test score of logistic regresion model on the test set:\n"
      f"{cv_results['test_score'].mean():.3f} +/- {cv_results['test_score'].std():.3f}")


# In[20]:


from sklearn.dummy import DummyClassifier
model = make_pipeline(StandardScaler(), DummyClassifier(strategy="most_frequent"))
cv_results = cross_validate(
    model, data, target, cv=10, return_estimator=True
)


# In[22]:


print(f"Test score of Dummy classifier on the test set:\n"
      f"{cv_results['test_score'].mean():.3f} +/- {cv_results['test_score'].std():.3f}")


# In[23]:


logistic_regression = make_pipeline(
    StandardScaler(), LogisticRegression(penalty="none")
)
logistic_regression.fit(data, target)


# In[28]:


coefs = logistic_regression[-1].coef_[0]  # the coefficients is a 2d array
weights = pd.Series(coefs, index=list(data.columns))


# In[29]:


weights.plot.barh()
plt.title("Weights of the logistic regression")


# In[30]:


model = make_pipeline(StandardScaler(), LogisticRegression())
cv_results = cross_validate(
    model, data, target, cv=10, return_estimator=True
)
coefs = [estimator[-1].coef_[0] for estimator in cv_results["estimator"]]
coefs = pd.DataFrame(coefs, columns=data.columns)
_, ax = plt.subplots(figsize=(15, 10))
coefs.abs().plot.box(**boxplot_property, ax=ax)


# In[31]:


adult_census = pd.read_csv("../datasets/adult-census.csv")
target = adult_census["class"]
data = adult_census.drop(columns=["class", "education-num"])


# In[32]:


data.info()


# In[33]:


data.isnull().sum()


# In[42]:


from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder

preprocessor = make_column_transformer(
    (OneHotEncoder(handle_unknown="ignore", sparse=False), selector(dtype_include=object)),
    (StandardScaler(), selector(dtype_exclude=object))
)

model = make_pipeline(preprocessor, LogisticRegression(max_iter=500))
cv_results = cross_validate(
    model, data, target, cv=10, return_estimator=True
)

cv_results['test_score'].mean()


# In[43]:


from sklearn.compose import make_column_selector as selector
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder

categorical_columns = selector(dtype_include=object)(data)
numerical_columns = selector(dtype_exclude=object)(data)

preprocessor = make_column_transformer(
    (OneHotEncoder(handle_unknown="ignore"), categorical_columns),
    (StandardScaler(), numerical_columns),
)
model = make_pipeline(preprocessor, LogisticRegression(max_iter=5000))
cv_results = cross_validate(
    model, data, target, cv=10, return_estimator=True, n_jobs=2
)
cv_results["test_score"].mean()


# In[44]:


preprocessor.fit(data)
feature_names = (preprocessor.named_transformers_["onehotencoder"]
                             .get_feature_names(categorical_columns)).tolist()
feature_names += numerical_columns


# In[45]:


coefs = [estimator[-1].coef_[0] for estimator in cv_results["estimator"]]
coefs = pd.DataFrame(coefs, columns=feature_names)
_, ax = plt.subplots(figsize=(15, 10))
coefs.abs().plot.box(**boxplot_property, ax=ax)


# In[49]:


coefs.abs().mean().sort_values()


# In[50]:


coefs = [estimator[-1].coef_[0] for estimator in cv_results["estimator"]]
coefs = pd.DataFrame(coefs, columns=feature_names)

# Define the style of the box style
boxplot_property = {
    "vert": False,
    "whis": 100,
    "patch_artist": True,
    "widths": 0.5,
    "boxprops": dict(linewidth=3, color="black", alpha=0.9),
    "medianprops": dict(linewidth=2.5, color="black", alpha=0.9),
    "whiskerprops": dict(linewidth=3, color="black", alpha=0.9),
    "capprops": dict(linewidth=3, color="black", alpha=0.9),
}

_, ax = plt.subplots(figsize=(10, 35))
_ = coefs.abs().plot.box(**boxplot_property, ax=ax)


# In[51]:


model = make_pipeline(
    preprocessor, LogisticRegression(C=0.01, max_iter=5000)
)
cv_results = cross_validate(
    model, data, target, cv=10, return_estimator=True, n_jobs=2
)
coefs = [estimator[-1].coef_[0] for estimator in cv_results["estimator"]]
coefs = pd.DataFrame(coefs, columns=feature_names)
_, ax = plt.subplots(figsize=(10, 35))
_ = coefs.abs().plot.box(**boxplot_property, ax=ax)


# In[ ]:




