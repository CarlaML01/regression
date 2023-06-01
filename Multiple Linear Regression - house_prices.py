#!/usr/bin/env python
# coding: utf-8

# ### Multiple Linear Regression with house prices
# 
# In this notebook, I have created a few simple linear regression models, as well as a multiple linear regression model, to predict home value.
# 
# Let's get started by importing the necessary libraries and reading in the data you will be using.

# In[1]:


import numpy as np
import pandas as pd
import statsmodels.api as sm;

df = pd.read_csv('./house_prices.csv')
df.head()


# `1.` Using statsmodels, fit three individual simple linear regression models to predict price.  You should have a model that uses **area**, another using **bedrooms**, and a final one using **bathrooms**.  You will also want to use an intercept in each of your three models.
# 
# Use the results from each of your models to answer the first two quiz questions below.

# In[7]:


# Fitting a simple linear regression model for area
X_area = df['area']
y = df['price']
X_area = sm.add_constant(X_area) # add an intercept
model_area = sm.OLS(y, X_area).fit()
print(model_area.summary())


# In[8]:


# Fitting a simple linear regression model for bedrooms
X_bedrooms = df['bedrooms']
X_bedrooms = sm.add_constant(X_bedrooms) # add an intercept
model_bedrooms = sm.OLS(y, X_bedrooms).fit()
print(model_bedrooms.summary())


# In[9]:


# Fitting a simple linear regression model for bathrooms
X_bathrooms = df['bathrooms']
X_bathrooms = sm.add_constant(X_bathrooms) # add an intercept
model_bathrooms = sm.OLS(y, X_bathrooms).fit()
print(model_bathrooms.summary())


# >Based on the results of the three simple linear regression models, each of the variables claim their significance in predicting price.

# `2.` Now that you have looked at the results from the simple linear regression models, let's try a multiple linear regression model using all three of these variables  at the same time.  You will still want an intercept in this model.

# In[10]:


df['intercept'] =1


# In[11]:


lm = sm.OLS(df['price'], df[['intercept','bathrooms', 'bedrooms', 'area']])
Results = lm.fit()
Results.summary()


# In[12]:


# Another way to Fit the multiple linear regression model using all three predictor variables
X = df[['area', 'bedrooms', 'bathrooms']]
X = sm.add_constant(X) # add an intercept
model_multiple = sm.OLS(y, X).fit()
print(model_multiple.summary())


# `3.` Along with using the **area**, **bedrooms**, and **bathrooms** you might also want to use **style** to predict the price.  Try adding this to your multiple linear regression model.  What happens?  Use the final quiz below to provide your answer.

# In[17]:


# Converting the categorical style variable into dummy variables
style_dummies = pd.get_dummies(df['style'], prefix='style')
style_dummies.head()


# In[18]:


# Concatenating the dummy variables with the original dataset
df = pd.concat([df, style_dummies], axis=1)
df.head()


# In[16]:


# Fitting a multiple linear regression model using all four predictor variables
X = df[['area', 'bedrooms', 'bathrooms', 'style_ranch', 'style_victorian']]
X = sm.add_constant(X) # add an intercept
model_multiple_style = sm.OLS(y, X).fit()
print(model_multiple_style.summary())


# >Using a multiple linear regression model shows only area as statistically significant.

# In[ ]:




