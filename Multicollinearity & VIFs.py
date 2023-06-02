#!/usr/bin/env python
# coding: utf-8

# ### Multicollinearity & VIFs
# 
# Using the notebook here, I will answer some questioons regarding multicollinearity.
# 
# To get started let's read in the necessary libraries and the data that will be used.

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from patsy import dmatrices
import statsmodels.api as sm;
from statsmodels.stats.outliers_influence import variance_inflation_factor
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('./house_prices.csv')
df.head()


#  

# `1.`Using [seaborn](https://seaborn.pydata.org/examples/scatterplot_matrix.html) to look at pairwise relationships for all of the quantitative, explanatory variables in the dataset by running the cell below.  

# In[2]:


sns.pairplot(df[['bedrooms', 'bathrooms', 'area']]);
#They all look positive correlations 


# >These x-variables all seem to have positive correlations with one another, which raises the problem of multicollinearity.

# `2.` Earlier, I have fitted linear models between each individual predictor variable and price, as well as used all of the variables and the price in a multiple linear regression model. Each of the individual models showed a **positive relationship** - that is, when bathrooms, bedrooms, or area increase, we predict the price of a home to increase. 
# 
# Fitting a linear model to predict a home **price** using **bedrooms**, **bathrooms**, and **area**. 

# In[2]:


y, X = dmatrices('price ~ bedrooms + bathrooms + area', data=df, return_type='dataframe')
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())


# >This code uses the dmatrices function from the patsy library to create a design matrix X and a response vector y from your dataset df. The sm.OLS function from the statsmodels library is then used to fit the linear model to the data, and the summary method is called on the results object to display a summary of the model.

# > **From the coefficients in our multiple linear regression model:**
#     - Contradictory to what you might imagine, when the number of bedrooms increases, we actually predict the home price to decrease. 
#     - The other variables have positive coefficients. The reason for this is because of multicollinearity.

# `3.` Calculating the VIFs for each variable in our model.

# In[3]:


vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
print(vif)


# > This code creates an empty dataframe vif, and then calculates the VIF for each variable in X using the variance_inflation_factor function from the statsmodels.stats.outliers_influence library. The results are then added to the vif dataframe and printed.

# > **VIFs** shows > 10 for bedrooms and bathrooms. 
#     - We should remove one of the variables, which will actually reduce the VIF for the other high VIF variable, as these variables are related with one another.

# `4.` Removing bathrooms from our above model.  Refitting the multiple linear regression model and re-computing the VIFs.  

# In[4]:


#using dmatrices function to esclude the bathrooms variable:
y, X = dmatrices('price ~ bedrooms + area', data=df, return_type='dataframe')
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())


# >This code creates a new design matrix X and response vector y that excludes the bathrooms variable, and then fits a new linear model to the data using the sm.OLS function. The summary of the new model is then printed.

# In[5]:


#Finally, to recompute the VIFs for the new model, I used the same code as before:
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
print(vif)


# > All VIFs are below 10 now and all coefficients are positive. According to r-square, the change didn't effect it's value, so:
#     - Removing the number of bathrooms didn't hurt the predictive power of the model, and still improved the interpretability of the coefficients.

# In[ ]:




