#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

blood_transfusion = pd.read_csv("../datasets/blood_transfusion.csv")
data = blood_transfusion.drop(columns="Class")
target = blood_transfusion["Class"]


# In[2]:


from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_validate

dummy_classifier = DummyClassifier(strategy="most_frequent")

scores = cross_validate(
    dummy_classifier, data, target, cv=10, 
    scoring=['accuracy'],
    return_estimator=True, n_jobs=2)
print(scores['test_accuracy'].mean())


# In[3]:


dummy = DummyClassifier(strategy="most_frequent")
cv_results = cross_validate(
    dummy, data, target, cv=10, scoring=["accuracy", "balanced_accuracy"]
)
print(f"Average accuracy: {cv_results['test_accuracy'].mean():.3f}")
print(f"Average balanced accuracy: "
      f"{cv_results['test_balanced_accuracy'].mean():.3f}")


# In[4]:


from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()
cv_results = cross_validate(
    tree, data, target, cv=10, scoring=["accuracy", "balanced_accuracy"]
)
print(f"Average accuracy: {cv_results['test_accuracy'].mean():.3f}")
print(f"Average balanced accuracy: "
      f"{cv_results['test_balanced_accuracy'].mean():.3f}")


# In[30]:


from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=300)
cv_results = cross_validate(
    forest, data, target, cv=10, scoring=["accuracy", "balanced_accuracy"]
)
print(f"Average accuracy: {cv_results['test_accuracy'].mean():.3f}")
print(f"Average balanced accuracy: "
      f"{cv_results['test_balanced_accuracy'].mean():.3f}")


# In[15]:


from sklearn.ensemble import GradientBoostingClassifier

forest = RandomForestClassifier(n_estimators=300)
gbooster = GradientBoostingClassifier(n_estimators=300)


# In[8]:


from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score

cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=1)

forest_cv_results = cross_validate(
    forest, data, target, cv=cv, scoring="balanced_accuracy"
)

gbooster_cv_results = cross_validate(
    forest, data, target, cv=cv, scoring="balanced_accuracy"
)


# In[23]:


forest_better = 0
booster_better = 0
equal = 0
for _ in range(10):
    forest = RandomForestClassifier(n_estimators=300)
    gbooster = GradientBoostingClassifier(n_estimators=300)
    forest_cv_results = cross_validate(
        forest, data, target, cv=10, scoring=["balanced_accuracy"]
    )

    gbooster_cv_results = cross_validate(
        gbooster, data, target, cv=10, scoring=["balanced_accuracy"]
    )
    if forest_cv_results['test_balanced_accuracy'].mean() > gbooster_cv_results['test_balanced_accuracy'].mean():
        forest_better += 1
    elif forest_cv_results['test_balanced_accuracy'].mean() < gbooster_cv_results['test_balanced_accuracy'].mean():
        booster_better += 1
    else:
        equal += 1
print(forest_better, booster_better, equal)


# In[24]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold

n_try = 10
scores_rf, scores_gbdt = [], []
for seed in range(n_try):
    cv = KFold(n_splits=10, shuffle=True, random_state=seed)

    rf = RandomForestClassifier(n_estimators=300, n_jobs=2)
    scores = cross_val_score(
        rf, data, target, cv=cv, scoring="balanced_accuracy", n_jobs=2
    )
    scores_rf.append(scores.mean())

    gbdt = GradientBoostingClassifier(n_estimators=300)
    scores = cross_val_score(
        gbdt, data, target, cv=cv, scoring="balanced_accuracy", n_jobs=2
    )
    scores_gbdt.append(scores.mean())

compare = [s_gbdt > s_rf for s_gbdt, s_rf in zip(scores_gbdt, scores_rf)]
sum(compare)


# In[31]:


from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

histogram_gradient_boosting = HistGradientBoostingClassifier(
    max_iter=1000, 
    early_stopping=True,
    random_state=0
)

cv_results = cross_validate(
    histogram_gradient_boosting, data, target, cv=10, scoring=["accuracy", "balanced_accuracy"], n_jobs=2
)

print(f"Average accuracy: {cv_results['test_accuracy'].mean():.3f}")
print(f"Average balanced accuracy: "
      f"{cv_results['test_balanced_accuracy'].mean():.3f}")


# In[29]:


gbdt = GradientBoostingClassifier(n_estimators=300)
    
cv_results = cross_validate(
    gbdt, data, target, cv=10, scoring=["accuracy", "balanced_accuracy"]
)

print(f"Average accuracy: {cv_results['test_accuracy'].mean():.3f}")
print(f"Average balanced accuracy: "
      f"{cv_results['test_balanced_accuracy'].mean():.3f}")


# In[33]:


hgbdt = HistGradientBoostingClassifier(
    max_iter=1000, 
    early_stopping=True,
    random_state=0
)
hgbdt.fit(data, target)
hgbdt.n_iter_


# In[36]:


import numpy as np

hgbdt = HistGradientBoostingClassifier(
    max_iter=1000, 
    early_stopping=True,
    random_state=0
)

cv_results = cross_validate(
    hgbdt, data, target, cv=10, scoring=["balanced_accuracy"], n_jobs=2, return_estimator=True
)

np.mean([estimator.n_iter_ for estimator in cv_results["estimator"]])


# In[37]:


from imblearn.ensemble import BalancedBaggingClassifier

bbc = BalancedBaggingClassifier(base_estimator=hgbdt,n_estimators=50)

cv_results = cross_validate(
    bbc, data, target, cv=10, scoring=["balanced_accuracy"], n_jobs=2, return_estimator=True
)

print(f"Average balanced accuracy: "
      f"{cv_results['test_balanced_accuracy'].mean():.3f}")


# In[39]:


from imblearn.ensemble import BalancedBaggingClassifier

balanced_bagging = BalancedBaggingClassifier(
    hgbdt, n_estimators=50, n_jobs=2, random_state=0
)
scores_balanced_bagging = cross_val_score(
    balanced_bagging, data, target, cv=10, scoring="balanced_accuracy",
    n_jobs=2
)
scores_balanced_bagging.mean()


# In[ ]:




