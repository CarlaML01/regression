#!/usr/bin/env python
# coding: utf-8

# ### Regression Carats vs. Price
# 
# In this notebook, you will perform a similar analysis to the one you did in the previous notebook, but using a dataset holding the weight of a diamond in carats, and the price of the corresponding diamond in dollars.
# 
# To get started, let's read in the necessary libraries and the dataset.

# In[2]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('./carats.csv', header= None)
df.columns = ['carats', 'price']
df.head()


# `1.` Similar to the last notebook, fit a simple linear regression model to predict price based on the weight of a diamond.  Use your results to answer the first question below.  Don't forget to add an intercept.

# In[3]:


# Define the predictor variable (carats) and the response variable (price)
X = df['carats']
y = df['price']

# Add a constant term to the predictor variable for the intercept term
X = sm.add_constant(X)

# Fit the linear regression model using OLS
model = sm.OLS(y, X).fit()

# Print the model summary to see the coefficients and other statistics
print(model.summary())


# `2.` Use [scatter](https://matplotlib.org/gallery/lines_bars_and_markers/scatter_symbol.html?highlight=scatter%20symbol) to create a scatterplot of the relationship between price and weight.  Then use the scatterplot and the output from your regression model to answer the second quiz question below.

# In[4]:


plt.scatter(df['carats'], df['price'])
plt.xlabel('Carats')
plt.ylabel('Price')
plt.show()


# In[5]:


#To fit a regression model to this data, you can use the OLS() function from the statsmodels library
X = df['carats']
y = df['price']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())


# In[6]:


#After you've created the scatterplot and fit the regression model, you can use the output from the model to plot a line of best fit on the scatterplot.
plt.scatter(df['carats'], df['price'])
plt.plot(df['carats'], model.predict(X), color='red')
plt.xlabel('Carats')
plt.ylabel('Price')
plt.show()


# In[ ]:




