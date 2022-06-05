#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_excel(r"C:\Users\User\Desktop\Chargebacks.xlsx")


# In[3]:


with pd.option_context("display.max_rows", None, "display.max_columns", None):
  display(df)


# In[4]:


df_explore = df.copy()


# # Data Hiccups Analysis

# In[5]:


df_explore.info()


# In[6]:


missing_cols, missing_rows = (
    (df_explore.isnull().sum(x) | df_explore.eq('').sum(x))
    .loc[lambda x: x.gt(0)].index
    for x in (0, 1)
)


# In[7]:


df_explore.loc[missing_rows, missing_cols]


# # Distribution of time to first payment by seller

# In[8]:


df_explore['DTFP'].median()


# In[9]:


df_explore['DTFP'].max()


# In[10]:


df_explore['DTFP'].min()


# In[11]:


df_explore['DTFP'].describe()


# In[12]:


plt.figure(figsize=(16,8))
sns.heatmap(df_explore.corr(), annot=True, linewidths=2, cmap= 'coolwarm')


# # percent of total gross payment volume (GPV) comes from sellers located in San Francisco

# In[13]:


df_explore['payment_size_usd'].sum()


# In[14]:


df1=df_explore['payment_size_usd'].sum()


# In[15]:


df_explore[df_explore['latitude']==37.7798928]['payment_size_usd'].sum()


# In[16]:


df2=df_explore[df_explore['latitude']==37.7798928]['payment_size_usd'].sum()


# In[17]:


print(df2/df1)


# # top three MCCs (business_category) by fraud chargeback rate (calculated using dollars)

# In[18]:


df_explore[df_explore['chargeback_type']=='fraud'].groupby(['business_category'])['chargeback_size_usd'].sum().sort_values(ascending=[False])


# # Metric to gauge the accuracy of our rules/models

# In[19]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris


# Sklearn has a wonderful function called accuracy_score for classification models, which we can use to test how accurate our model is. Essentially this is comparing the predicted values against the actual values in our model.
# 

# # evaluate accuracy
# print(accuracy_score(y_test, pred))

# # Number of correct predictions
# 
# comparison = (y_test == pred)
# np.count_nonzero(comparison) #although its called non_zero, with boolean values this tests for # of True

# comparison

# # the number of mispredictions
# 
# np.size(comparison)-np.count_nonzero(comparison)

# # Same result as above - the number of mispredictions
# 
# np.count_nonzero(comparison == False)

# Now, with classification models like this one, our accuracy will change with different values of neighbours. We are going to build a loop to test different how different values of K impact our model. First we will generate 10 random train/test splits for each value of k, and then we will calculate the average accuracy for each value.

# num_splits = 10 #train test splits
# k_vals = [1, 3, 5, 7, 10, 20, 30, 40, 50] #each k value to be tested
# accuracies = [] #list to collect the accuracy scores
# comparisons = [] #list to collect the number of non_zero (or TRUE) values
# 
# for k in k_vals: #for each value of k we are going to first create an accuracy list, and a comp list, then populate these lists
#     acc_samples = []
#     comp_samples = []
#     for i in range(num_splits):
#         # make sure we don't set the `random_state` parameter to an integer, we want random splits
#         X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
#         knn = KNeighborsClassifier(n_neighbors=k)
#         knn.fit(X_train, y_train)
#         y_pred = knn.predict(X_test)
#         acc_samples.append(accuracy_score(y_test, y_pred)) #calculate the accuracy score and add to list
#         comp_samples.append(np.count_nonzero(y_test == y_pred)) #count non_zeros and add to list
#     accuracies.append(np.mean(acc_samples)) #for each k's list, calculate the mean score and add to master list
#     comparisons.append(np.mean(comp_samples)) #for each k's list, calculate the mean non_zero values & add to master list
# 
#     #create a dataframe
# compare_df = pd.DataFrame({'Correct Predictions':comparisons,'Accuracy Score':accuracies}, index=[k_vals])
# compare_df

# Plot the dataframe values

# #4.1 Manual Review by Biz Category

# In[25]:


df_explore[df_explore['chargeback_type']=='fraud'].groupby(['business_category'])['was_cased'].count()


# In[33]:


df3=df_explore.groupby(['business_category','was_cased'])['business_category'].count().unstack()
df3


# In[36]:


df3['total_cases']=df3[0]+df3[1]


# In[40]:


df3['%_review']=df3[1]/df3['total_cases']
df3


# In[41]:


df3['%_review'].sort_values


# #4.2 Fraud Chargeback % by Category not by dollar amount

# In[42]:


df4=df_explore.groupby(['business_category','chargeback_type'])['business_category'].count().unstack()
df4


# In[44]:


df4['total_cases']=df4['fraud']+df4['non-fraud']


# In[46]:


df4['%fraud']=df4['fraud']/df4['total_cases']
df4


# In[48]:


df4['%fraud'].sort_values(ascending=[False])


# In[ ]:




