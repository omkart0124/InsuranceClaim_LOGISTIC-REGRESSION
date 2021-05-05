#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


# one thing always have to do with regards to classification :

# check the data is balanced 
# meand our target(insuranceclaim) that is equally devided because it is a binary classification that means both (0 and 1) close to each other
# when we say balance we have to focus only on the target


# In[3]:


# what if data is not balanced(interview question)
# if data is not balance we have to filter out information


# In[ ]:





# In[ ]:





# In[4]:


d= pd.read_csv(r"D:\ML\insurance.csv")


# In[5]:


d


# In[42]:


d.corr()
# rest of on x axis
# insuranceclaim is our target


# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[8]:


plt.figure(figsize=(8,8))
sns.heatmap(d.corr(), annot=True)


# In[9]:


d.info()
# no null values
# all data in int and float


# In[10]:


d['insuranceclaim'].unique()


# In[11]:


# there is no junk or nan values in the data


# In[12]:


d.head()


# In[ ]:





# In[ ]:





# In[13]:


# check if te data is balanced (target) 
d['insuranceclaim'].value_counts()


# In[14]:


d['insuranceclaim'].value_counts().plot(kind="bar")
# here we can see there is a slight diff but not tomuch of diff
# bcause 1 : 783 and 0 is 555
# there is not much diff so we can declared that this data set is balcanced


# In[15]:


from sklearn.model_selection import train_test_split
# x = d.drop("insuranceclaim", axix = 1)-----this is one to drop
x = d.iloc[:, :-1]    # it means from 0 to -1 pick the data 
y = d.iloc[:, -1]     # take only -1
# basically we devide x and y


# In[16]:


x.head()
# here we can see the data is without the insuranceclaim


# In[17]:


y.head()
# here we only get insuranceclaim


# In[ ]:





# In[18]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=41)

# random_state is a input to randomize your split


# In[19]:


from sklearn.linear_model import LogisticRegression


# In[20]:


log = LogisticRegression()
log.fit(x_train,y_train)
print(log.coef_)
print(log.intercept_)


# In[21]:


log.predict_proba(x_test) #    doing this we get prob of x

# PREDICT PROBA is use for the prob value of the number
# and when your prob value are greater than 0.5 the ans you get in the predict is 1
# because  its going to make split based on thresh-hold values

#if the answer is <0.5 your value is going to be 0


# In[22]:


log.predict(x_test)   # predict is just instance values that gives 0 and 1


# In[23]:


y_hat = log.predict(x_test)  # storing values into y_hat 


# In[24]:


from sklearn.metrics import confusion_matrix
# Classification accuracy alone can be misleading if you have an unequal number of observations
# in each class or if you have more than two classes in your dataset.


# In[25]:


confusion_matrix(y_test,y_hat)

# 119 is true(-ve)
# 54 false(+ve)
# 33 false (-ve)
# 196 true(+ve )


# In[26]:


# now here we are going to store above values into variable for that jst follow order


# In[27]:


tn, fp, fn, tp = confusion_matrix(y_test,y_hat).ravel()

# ravel means flatten the 2d array
# we can also use flatten keyword or ravel both are same


# In[28]:


print(tp, fn)
print(fp, tn)


# In[29]:


from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score)


# In[30]:


accuracy_score(y_test,y_hat)


# In[31]:


precision_score(y_test,y_hat)


# In[32]:


recall_score(y_test,y_hat)


# In[33]:


f1_score(y_test, y_hat)


# In[ ]:





# In[34]:


# ROC
# we are going to draw ROC curve to check how the output is going to be like


# In[35]:


from sklearn.metrics import roc_curve, roc_auc_score


# In[36]:


roc_curve(y_test, y_hat)                                              #1st fpr, 2nd tpr, 3rd threshold value always 2,1,0


# In[37]:


fpr, tpr, thres = roc_curve(y_test,y_hat)

# add arrays into variable names
# doing for line plot
# And these line plot specifically design perspective of above values


# In[38]:


best_fpr=[0,0,1]
best_tpr=[0,1,1]

worst_fpr=[0,1]
worst_tpr=[0,1]


# In[39]:


plt.plot(worst_fpr, worst_tpr, "r-",label="worst")
plt.plot(fpr,tpr, 'b-',label='Current')
plt.plot(best_fpr,best_tpr, 'g-',label='best')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()


# red is worst because its just cover 50% of the data
# green is our data point 
# 0to 1 and till green line is our area under  curve (AUC)
# 


# In[40]:


# to check how much data is covered under the AUC
roc_auc_score(y_test,y_hat)


# In[41]:


# 77% DATA IS COVERED UNDER THE AUC and Puc cure 

