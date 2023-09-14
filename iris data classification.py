#!/usr/bin/env python
# coding: utf-8

# In[39]:


import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[40]:


sujal_csv=pd.read_csv("Iris.csv")
(sujal_csv)


# In[41]:


sujal_csv.isnull().sum()


# In[42]:


sujal_csv.head(10)


# In[43]:


sujal_csv.tail()


# In[44]:


sujal_csv.describe()


# In[45]:


sujal_csv['Species'].unique()


# In[46]:


#dataset information 
sujal_csv.info()


# In[47]:


# visuallize the dataset using the seaborn library 
sns.pairplot(sujal_csv, data='Label')


# In[48]:


a=sujal_csv.values
X =a[:,0:4]
Y =a[:,4]


# In[49]:


# train the dataset 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


# In[50]:


#Applying the linear regression 
model= LinearRegression()


# model.fit(X,Y)

# In[58]:


model.fit(X_train, y_train)
print(model.score(X_test, y_test))


# In[59]:


#model intercept 
model.intercept_


# In[60]:


# model coefficient 
model.coef_


# In[62]:


# predict the model 
y_pred=model.predict(X_test)


# In[63]:


# mean squared error
np.mean((y_pred-y_test)**2)


# In[ ]:




