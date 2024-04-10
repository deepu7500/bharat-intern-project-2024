#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn import metrics


# In[2]:


from sklearn.linear_model import LinearRegression


# In[3]:


from sklearn.tree import DecisionTreeClassifier


# In[4]:


import seaborn as sns


# In[5]:


iris= sns.load_dataset('iris')


# In[6]:


iris


# In[ ]:





# In[7]:


iris.head


# In[8]:


iris.describe()


# In[9]:


from sklearn.model_selection import train_test_split
train,test=train_test_split(iris,test_size=0.25)


# In[10]:


train


# In[11]:


test


# In[12]:


train_x=train[['sepal_length','sepal_width','petal_length','petal_width']]


# In[13]:


train_y=train.species


# In[14]:


train_x


# In[15]:


train_y


# In[16]:


test_x=test[['sepal_length','sepal_width','petal_length','petal_width']]


# In[17]:


test_y=test.species


# In[18]:


test_x


# In[19]:


test_y


# In[20]:


from sklearn import svm
model=svm.SVC()


# In[21]:


model


# In[22]:


model.fit(train_x,train_y)
pred=model.predict(test_x)


# In[23]:


pred


# In[24]:


metrics.accuracy_score(pred,test_y)


# In[25]:


model=DecisionTreeClassifier()


# In[26]:


model.fit(train_x,train_y)


# In[27]:


pred 


# In[28]:


pred=model.predict(test_x)


# In[29]:


metrics.accuracy_score(pred,test_y)


# In[30]:


iris.columns


# In[31]:


iris.species.nunique()


# In[35]:


import matplotlib.pyplot  as plt
sns.boxplot(x='species',y='petal_length',data=iris)
plt.show()


# In[37]:


sns.boxplot(x='species',y='sepal_width',data=iris)


# In[38]:


sns.boxplot(x='species',y='sepal_width',data=iris)


# In[39]:


sns.boxplot(x='species',y='sepal_length',data=iris)


# In[40]:


sns.boxplot(x='species',y='petal_width',data=iris)


# In[41]:


sns.boxplot(y='sepal_length', data=iris)


# In[43]:


sns.pairplot(iris,hue='species')


# In[ ]:





# In[ ]:




