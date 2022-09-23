#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[3]:


df=sns.load_dataset('tips')
df


# In[6]:


flights=sns.load_dataset('flights')
flights


# In[7]:


iris=sns.load_dataset('iris')
iris


# In[20]:


url='https://raw.githubusercontent.com/campusx-official/100-days-of-machine-learning/main/day21-bivariate-analysis/train.csv'
titanic=pd.read_csv(url)
titanic.sample(20)


# In[ ]:





# # scatter plot (numarical- numarical)
# 

# In[23]:


tips=sns.load_dataset('tips')
tips.head()


# In[40]:


sns.scatterplot(tips['total_bill'],tips['tip'],hue=tips['sex'],style=tips['smoker'],size=tips['size'])


# # bar chart numarical -categorical

# In[42]:


url='https://raw.githubusercontent.com/campusx-official/100-days-of-machine-learning/main/day21-bivariate-analysis/train.csv'
titanic=pd.read_csv(url)
titanic.sample(5)


# In[66]:


sns.barplot(titanic['Pclass'],titanic['Age'])


# In[62]:


sns.barplot(titanic['Pclass'],titanic['Survived'],hue=titanic['Sex'])


# In[59]:


sns.barplot(titanic['Sex'],titanic['Survived'])


# In[61]:


sns.barplot(titanic['Age'],titanic['Survived'],hue=titanic['Sex'])


# # box plot (numarical -categorical)

# In[69]:


sns.boxplot(titanic['Sex'],titanic['Age'])


# # distplot (numarical -categorical)

# In[81]:


sns.distplot(titanic[titanic['Survived']==0]['Age'],hist=False)
sns.distplot(titanic[titanic['Survived']==1]['Age'],hist=False)


# In[87]:


pd.crosstab(titanic['Pclass'],titanic['Survived'])


# # Heatmap (categorical- categorical)

# In[89]:


sns.heatmap(pd.crosstab(titanic['Pclass'],titanic['Survived']))


# In[96]:


titanic.groupby('Pclass').mean()['Survived']*100


# In[102]:


(titanic.groupby('Pclass').mean()['Survived']*100).plot(kind='bar')


# # clustermap (categorical- categorical)

# In[105]:


sns.clustermap(pd.crosstab(titanic['SibSp'],titanic['Survived']))


# In[106]:


sns.clustermap(pd.crosstab(titanic['Parch'],titanic['Survived']))


# # Pairplot (different relationship)

# In[107]:


iris.head()


# In[109]:


sns.pairplot(iris)# ralation with all column to each-other 


# In[111]:


sns.pairplot(iris,hue='species')


# # line plot use for numerical to numarical (specially use for time series)

# In[113]:


flights.head(10)


# In[120]:


time_series=flights.groupby('year').sum().reset_index()


# In[121]:


sns.lineplot(time_series['year'],time_series['passengers'])


# In[ ]:




