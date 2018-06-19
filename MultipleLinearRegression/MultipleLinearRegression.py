
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import scipy.stats as stats
import sklearn
import matplotlib.pyplot as plt


# In[2]:


from sklearn.datasets import load_boston
boston=load_boston()


# In[3]:


boston.keys()


# In[4]:


boston.data.shape


# In[5]:


boston.feature_names


# In[6]:


print boston.DESCR


# In[23]:


bos=pd.DataFrame(boston.data)


# In[24]:


bos.head()


# In[25]:


bos.columns=boston.feature_names


# In[26]:


bos.head()


# In[32]:


bos['PRICE']=boston.target


# In[36]:



bos.head()
X=bos.drop('PRICE',axis=1)


# In[57]:


from sklearn.linear_model import LinearRegression
lm=LinearRegression()


# In[63]:


X_train, X_test,Y_train, Y_test=sklearn.model_selection.train_test_split(X,bos.PRICE,test_size=0.33)


# In[65]:


print X_train.shape, Y_train.shape
print X_test.shape, Y_test.shape
lm.fit(X_train,Y_train)


# In[66]:


predictions=lm.predict(X_test)


# In[68]:


plt.scatter(Y_test, predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()


# In[71]:


lm.score(X_test,Y_test)


# In[74]:


meanSquaredError=np.mean((predictions-Y_test)**2)
print meanSquaredError

