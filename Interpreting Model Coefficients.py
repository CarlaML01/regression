#!/usr/bin/env python
# coding: utf-8

# ### Interpreting Coefficients
# 
# In this notebook, I will fit a few different models.
# In some cases, the coefficients of our linear regression models wouldn't be kept due to the lack of significance. But that is not the aim of this notebook - **this notebook is strictly to assure I am comfortable with how to interpret coefficients when they are interpretable at all**.

# In[11]:


import numpy as np
import pandas as pd
import statsmodels.api as sm;

df = pd.read_csv('./house_prices.csv')
df.head()


# We will be fitting a number of different models to this dataset throughout this notebook. 
# 
# ### Model 1
# 
# `1.` For the first model, let's fit a model to predict `price` using `neighborhood`, `style`, and the `area` of the home.  Using the output to match the correct values to the corresponding interpretation.  

# In[12]:


#Let's create dummy variables for the categorical features 'neighborhood' and 'style'. We'll drop the first category to avoid multicollinearity.
# Creating dummy variables
neighborhood_dummies = pd.get_dummies(df['neighborhood'], prefix='neighborhood', drop_first=True)
style_dummies = pd.get_dummies(df['style'], prefix='style', drop_first=True)

# Adding dummy variables to the dataframe
df = pd.concat([df, neighborhood_dummies, style_dummies], axis=1)

# Dropping original 'neighborhood' and 'style' columns
df.drop(['neighborhood', 'style'], axis=1, inplace=True)
df.head()


# In[13]:


#Now, let's build the first model using 'neighborhood_B', 'neighborhood_C', 'style_ranch', 'style_victorian', and 'area' as predictors.
# Defining the predictors and the target variable
X1 = df[['neighborhood_B', 'neighborhood_C', 'style_ranch', 'style_victorian', 'area']]
y = df['price']

# Adding an intercept to the predictors
X1 = sm.add_constant(X1)

# Fitting the model
model1 = sm.OLS(y, X1).fit()
print(model1.summary())


# >Turns out we can interpret all of the coefficients in this first model. Since there are no higher order terms

# > - The predicted difference in price between a victorian and lodge home, holding all other variables constant is more expensive by 6262.73.
# > - The predicted difference in the price of a home in neighborhood in A as compared to neighborhood C, holding other variables constant: -194.25
# > - For every one unit increase in the area of a home, we predict the price of the home to increase by 348.74 (holding all other variables constant)?
# > - The predicted home price if the home is a lodge in neighborhood C with an area of 0: -198300
# 

# ### Model 2
# 
# `2.` Now let's try a second model for predicting price.  This time, using `area` and `area squared` to predict price.  Also using the `style` of the home, but not `neighborhood` this time. I will again need to use your dummy variables, and add an intercept to the model. 

# In[14]:


#First, let's create the 'area_squared' feature.

# Creating 'area_squared' feature
df['area_squared'] = df['area'] ** 2


# In[15]:


#adding an intercept to the dataframe
df['intercept'] = 1


# In[17]:


#Creating the multiple linear regression model using the 'area', 'area_squared', and dummy variables for 'style' as predictor variables
X = df[['intercept', 'area', 'area_squared', 'style_ranch', 'style_victorian']]
y = df['price']

model = sm.OLS(y, X)
results = model.fit()
print(results.summary())


# >With the higher order term, the coefficients associated with area and area squared are not easily interpretable. However, coefficients that are not associated with the higher order terms are still interpretable

# > Judging by the first results from the two models I built, the best would likely involve only these two variables, as it would be simplified, while still predicting well.

# In[ ]:




