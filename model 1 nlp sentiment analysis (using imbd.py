#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
temp_df = pd.read_csv("IMDB Dataset.xls")
temp_df


# In[2]:


temp_df.shape


# In[3]:


temp_df["review"][1]


# In[4]:


temp_df.nunique()


# In[5]:


df=temp_df.iloc[:10000]
df


# In[6]:


df["sentiment"].value_counts()


# In[7]:


df.isnull().sum()


# In[8]:


df.duplicated().sum()


# In[9]:


# dropping duplicates
df.drop_duplicates(inplace=True)


# In[10]:


# agin check duplicates for verfication
df.duplicated().sum()


# In[11]:


import re
def remove_tags(raw_text):
    cleaned_text = re.sub(re.compile('<.*?>'),  ' ', raw_text)
    return cleaned_text


# In[12]:


df['review'] = df['review'].apply(remove_tags)


# In[13]:


df


# In[14]:


df['review'] = df['review'].apply(lambda x:x.lower())


# In[15]:


df


# In[16]:


from nltk.corpus import stopwords

sw_list = stopwords.words('english')

df['review'] = df['review'].apply(lambda x: [item for item in x.split() if item not in sw_list]).apply(lambda x:" ".join(x))


# In[19]:


df


# In[20]:


X = df.iloc[:,0:1]
y = df['sentiment']


# In[21]:


X


# In[22]:


y


# In[23]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(y)


# In[24]:


y


# In[25]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# In[26]:


X_train.shape


# In[27]:


print(X_train.shape)
print(X_test.shape)
print(y_test.shape)
print(y_train.shape)


# In[28]:


# Applying BOW
from sklearn.feature_extraction.text import CountVectorizer


# In[29]:


cv = CountVectorizer()


# In[30]:


X_train_bow = cv.fit_transform(X_train['review']).toarray()
X_test_bow = cv.transform(X_test['review']).toarray()

# Vimp ) transform only train data ko krte hai.


# In[31]:


X_train_bow.shape


# In[32]:


X_test_bow.shape


# In[33]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train_bow, y_train)


# In[34]:


y_pred = gnb.predict(X_test_bow)


# In[35]:


from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(y_test, y_pred)


# In[36]:


confusion_matrix(y_test,y_pred)


# In[37]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train_bow, y_train)
y_pred = rf.predict(X_test_bow)


# In[38]:


accuracy_score(y_test, y_pred)


# In[39]:


cv = CountVectorizer(max_features=3000)


# In[40]:


X_train_bow = cv.fit_transform(X_train['review']).toarray()
X_test_bow = cv.transform(X_test['review']).toarray()
rf = RandomForestClassifier()
rf.fit(X_train_bow, y_train)
y_pred = rf.predict(X_test_bow)


# In[41]:


accuracy_score(y_test, y_pred)


# In[43]:


cv = CountVectorizer(ngram_range=(1,2),max_features=5000)
X_train_bow = cv.fit_transform(X_train['review']).toarray()
X_test_bow = cv.transform(X_test['review']).toarray()
rf = RandomForestClassifier()
rf.fit(X_train_bow, y_train)
y_pred = rf.predict(X_test_bow)
accuracy_score(y_test, y_pred)


# # using Tfidf

# In[44]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[45]:


tfidf = TfidfVectorizer()


# In[46]:


X_train_tfidf = tfidf.fit_transform(X_train['review']).toarray()
X_test_tfidf = tfidf.transform(X_test['review'])


# In[ ]:


rf = RandomForestClassifier()
rf.fit(X_train_tfidf, y_train)
y_pred = rf.predict(X_test_tfidf)
accuracy_score(y_test, y_pred)


# In[ ]:


confusion_matrix(y_test,y_pred)


# In[ ]:


# Step 1: Input the new review
new_review = ["this is a very bad movie"]

# Step 2: Transform the review using TF-IDF
new_review_tfidf = tfidf.transform(new_review)

# Step 3: Predict sentiment
predicted_sentiment = rf.predict(new_review_tfidf)

# Step 4: Print the result
if predicted_sentiment[0] == 1:
    print("Positive Sentiment")
else:
    print("Negative Sentiment")

    
# also try this is very good movie
# and interesting movie


# In[ ]:


predicted_sentiment


# In[ ]:




