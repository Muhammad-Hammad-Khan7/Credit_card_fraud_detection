#!/usr/bin/env python
# coding: utf-8

# # Part 1: Data preprocessing

# Dataset link: https://www.kaggle.com/mlg-ulb/creditcardfraud?select=creditcard.csv

# ## Importing the libraries and the dataset

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# In[ ]:


# Install kaggle API
! pip install -q kaggle

# In[ ]:


# upload kaggle API key to colab notebook

# In[ ]:


# Make directory named kaggle and copy kaggle.json file there
! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/

# In[ ]:


# disable API key
! chmod 600 /root/.kaggle/kaggle.json

# In[ ]:


# import the dataset
! kaggle datasets download -d mlg-ulb/creditcardfraud

# In[ ]:


# unzipping the dataset
! unzip -q /content/creditcardfraud.zip

# ## Data Exploration

# In[ ]:


dataset = pd.read_csv('/content/creditcard.csv')

# In[ ]:


dataset.head()

# In[ ]:


dataset.shape

# In[ ]:


dataset.columns

# In[ ]:


dataset.info()

# In[ ]:


# statistical summary
dataset.describe()

# ## Dealing with missing values

# In[ ]:


dataset.isnull().values.any()

# In[ ]:


dataset.isnull().values.sum()

# ## Encoding categorical data

# In[ ]:


dataset.select_dtypes(include='object').columns

# In[ ]:


len(dataset.select_dtypes(include='object').columns)

# ## Countplot

# In[ ]:


sns.countplot(dataset['Class'])

# In[ ]:


# non fraud transactions
(dataset.Class == 0).sum()

# In[ ]:


# fraud transactions
(dataset.Class == 1).sum()

# ## Correlation matrix and heatmap

# In[ ]:


dataset_2 = dataset.drop(columns='Class')

# In[ ]:


dataset_2.corrwith(dataset['Class']).plot.bar(
    figsize=(16, 9), title='Correlated with Class', grid=True
)

# In[ ]:


corr = dataset.corr()

# In[ ]:


plt.figure(figsize=(16, 9))
ax = sns.heatmap(corr, annot=True, linewidths=2)

# ## Splitting the dataset

# In[ ]:


dataset.head()

# In[ ]:


# matrix of features / independent variables
x = dataset.drop(columns='Class')

# In[ ]:


# target variable / dependent variable
y = dataset['Class']

# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0)

# In[ ]:


x_train.shape

# In[ ]:


y_train.shape

# In[ ]:


x_test.shape

# In[ ]:


y_test.shape

# ## Feature scaling

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# In[ ]:


x_train

# In[ ]:


x_test

# # Part 2: Building the model

# ## 1) Logistic regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
classifier_lr = LogisticRegression(random_state=0)
classifier_lr.fit(x_train, y_train)

# In[ ]:


y_pred = classifier_lr.predict(x_test)

# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score

# In[ ]:


acc = accuracy_score(y_test, y_pred)
print(acc*100)

# In[ ]:


cm = confusion_matrix(y_test, y_pred)
print(cm)

# ## 2) Random forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(random_state=0)
classifier_rf.fit(x_train, y_train)

# In[ ]:


y_pred = classifier_rf.predict(x_test)

# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score
acc = accuracy_score(y_test, y_pred)
print(acc*100)

# In[ ]:


cm = confusion_matrix(y_test, y_pred)
print(cm)

# ## 3) XGBoost classifier

# In[ ]:


from xgboost import XGBClassifier
classifier_xgb = XGBClassifier(random_state=0)
classifier_xgb.fit(x_train, y_train)

# In[ ]:


y_pred = classifier_xgb.predict(x_test)

# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score
acc = accuracy_score(y_test, y_pred)
print(acc*100)

# In[ ]:


cm = confusion_matrix(y_test, y_pred)
print(cm)

# # Part 3: Final model (Random forest)

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(random_state=0)
classifier.fit(x_train, y_train)

# In[ ]:


y_pred = classifier.predict(x_test)

# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score
acc = accuracy_score(y_test, y_pred)
print(acc*100)

# In[ ]:


cm = confusion_matrix(y_test, y_pred)
print(cm)

# # Part 4: Predicting a single observation

# In[ ]:


dataset.head()

# In[ ]:


dataset.shape

# In[ ]:


single_obs = [[0.0, -1.359807,	-0.072781,	2.536347,	1.378155,	-0.338321,	0.462388,	0.239599,	0.098698,	0.363787,	0.090794,	-0.551600,	-0.617801,	-0.991390,	-0.311169,	1.468177,	-0.470401,	0.207971,	0.025791,	0.403993,	0.251412,	-0.018307,	0.277838,	-0.110474,	0.066928,	0.128539,	-0.189115,	0.133558,	-0.021053,	149.62
]]

# In[ ]:


classifier.predict(sc.transform(single_obs))
