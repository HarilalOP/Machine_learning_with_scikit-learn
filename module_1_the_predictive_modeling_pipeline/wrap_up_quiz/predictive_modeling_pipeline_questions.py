#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
ames_housing = pd.read_csv("../datasets/house_prices.csv", na_values="?")
ames_housing = ames_housing.drop(columns="Id")

target_name = "SalePrice"
data, target = ames_housing.drop(columns=target_name), ames_housing[target_name]
target = (target > 200_000).astype(int)


# In[2]:


ames_housing.head()


# In[3]:


data.head()


# In[4]:


target.head()


# In[5]:


data.info()


# In[6]:


numerical_features = [
  "LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2",
  "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF",
  "GrLivArea", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces",
  "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
  "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal",
]


# In[7]:


data_numeric = data[numerical_features]
data_numeric.head()


# In[9]:


from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

model = make_pipeline(SimpleImputer(strategy="mean"), 
                      StandardScaler(), 
                      LogisticRegression())


# In[10]:


model.fit(data_numeric, target)


# In[11]:


from sklearn.model_selection import cross_validate
cv_results = cross_validate(model, data_numeric, target, cv=5)
scores = cv_results["test_score"]
print(f"The accuracy is: {scores.mean():.1f} +/- {scores.std():.3f}")


# In[14]:


data.columns


# In[17]:


categorical_features = list(set(data.columns) -  set(numerical_features))


# In[22]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

scaler_imputer_transformer = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler())
categorical_imputer_transformer = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore"))

preprocessor = ColumnTransformer(transformers=[
    ("num_preprocessor", scaler_imputer_transformer, numerical_features),
    ("cat_preprocessor", categorical_imputer_transformer, categorical_features)
])

model = make_pipeline(preprocessor, LogisticRegression(max_iter=500))


# In[23]:


model.fit(data, target)


# In[24]:


from sklearn.model_selection import cross_validate
cv_results = cross_validate(model, data, target, cv=5)
scores = cv_results["test_score"]
print(f"The accuracy is: {scores.mean():.1f} +/- {scores.std():.3f}")


# In[ ]:




