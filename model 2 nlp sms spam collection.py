#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("smsspamcollection.tsv",sep="\t")


# In[3]:


df.head()


# In[4]:


df.head()


# In[6]:


df.describe()


# In[7]:


df['label'].value_counts()


# # Balancing the data

# In[8]:


# select ham data
ham = df[df['label']=='ham']
ham.head()

# taking all ham data in new ham variable.


# In[9]:


spam = df[df['label']=='spam']
spam.head()


# In[10]:


ham.shape, spam.shape


# In[11]:


spam.shape[0]


# In[12]:


ham=ham.sample(spam.shape[0])


# In[13]:


# check the shape of data
ham.shape, spam.shape


# In[14]:


data = pd.concat([ham,spam],ignore_index=True)


# In[15]:


data.shape


# In[16]:


data


# # Data Visualization

# In[18]:


# plot histogram of length for ham messages
plt.hist(data[data['label'] == 'ham']['length'], bins=100, alpha=0.7)
plt.show()
# from the histogram we can say that, the number of characters in ham messages are less than 1)


# In[19]:


plt.hist(data[data['label']=='ham']['length'],bins=100,alpha=0.7)
plt.hist(data[data['label']=='spam']['length'],bins=100,alpha=0.7)
plt.show()


# In[20]:


plt.hist(data[data['label']=='ham']['punct'],bins=100,alpha=0.7)
plt.hist(data[data['label']=='spam']['punct'],bins=100,alpha=0.7)
plt.show()


# In[21]:


from sklearn.model_selection import train_test_split


# In[22]:


x_train, x_test, y_train, y_test = train_test_split(data['message'],data['label'],test_size=0.3,
                                                  
                                                                   random_state=0, shuffle=True)


# In[23]:


from sklearn.pipeline import Pipeline
# there will be lot of repeated processes for training and testing the dataset separately,
# to avoid that we are using pipeline

from sklearn.feature_extraction.text import TfidfVectorizer
# we are importing TfidfVectorizer to utilize bag of words model in sklearn

from sklearn.ensemble import RandomForestClassifier


# In[24]:


classifier = Pipeline([('tfidf', TfidfVectorizer()), ('classifier',RandomForestClassifier(n_estimators=100))])


# In[25]:


classifier.fit(x_train, y_train)


# # Predicting the result(Random Forest)

# In[26]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[27]:


y_pred = classifier.predict(x_test)


# In[28]:


# confusion_matrix
confusion_matrix(y_test, y_pred)


# In[29]:


print(classification_report(y_test,y_pred))


# In[30]:


accuracy_score(y_test, y_pred)


# In[31]:


# Predict a real message
classifier.predict(['Hello, You are learning natural language Processing'])


# In[33]:


classifier.predict(['Hope you are doing good and learning new things !'])


# In[34]:


classifier.predict(['Congraturation, you won 50 crore'])


# In[ ]:




