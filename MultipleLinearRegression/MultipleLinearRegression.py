
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import scipy.stats as stats
import sklearn
import matplotlib.pyplot as plt


# In[3]:


from sklearn.datasets import load_boston
boston=load_boston()


# In[4]:


boston.keys()


# In[5]:


boston.data.shape


# In[6]:


boston.feature_names


# In[7]:


print boston.DESCR


# In[8]:


bos=pd.DataFrame(boston.data)


# In[9]:


bos.head()


# In[10]:


bos.columns=boston.feature_names


# In[11]:


bos.head()


# In[12]:


bos['PRICE']=boston.target


# In[13]:



bos.head()
X=bos.drop('PRICE',axis=1)


# In[14]:


from sklearn.linear_model import LinearRegression
lm=LinearRegression()


# In[15]:


X_train, X_test,Y_train, Y_test=sklearn.model_selection.train_test_split(X,bos.PRICE,test_size=0.33)


# In[16]:


print X_train.shape, Y_train.shape
print X_test.shape, Y_test.shape
lm.fit(X_train,Y_train)


# In[24]:


predictions=lm.predict(X_test)


# In[28]:


plt.scatter(Y_test, predictions)
plt.plot(Y_test,Y_test)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()


# In[20]:


score=lm.score(X_test,Y_test)
print "Score:",score


# In[21]:


meanSquaredError=np.mean((predictions-Y_test)**2)
print meanSquaredError


# In[22]:


#Performing backward elimination to improve predictions
import statsmodels.formula.api as sm
X=np.append(arr=np.ones((506,1)).astype(int),values=X,axis=1)
X_opt=X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13]]
regressor_OLS=sm.OLS(endog=bos.PRICE,exog=X_opt).fit()
regressor_OLS.summary()


# In[23]:


X_opt=X[:,[0,1,2,3,4,5,6,8,9,10,11,12,13]]
regressor_OLS=sm.OLS(endog=bos.PRICE,exog=X_opt).fit()
regressor_OLS.summary()


# In[29]:


X_opt=X[:,[0,1,2,4,5,6,8,9,10,11,12,13]]
regressor_OLS=sm.OLS(endog=bos.PRICE,exog=X_opt).fit()
regressor_OLS.summary()


# In[38]:


lm.fit(X_opt,bos.PRICE)
opt_score=lm.score(X_opt,bos.PRICE)
print "Score after backward elimination:",opt_score


# In[35]:




