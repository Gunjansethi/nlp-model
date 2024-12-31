#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv("smsspamcollection.tsv",sep="\t")
df.head()


# In[3]:


df.sample(5)


# In[4]:


df.shape


# # data cleaning

# In[5]:


df.isnull().sum()


# In[6]:


df.info()


# In[7]:


df1 = df.drop(columns=['length', 'punct'])


# In[8]:


df1.sample(5)


# In[9]:


# Label Encoder

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[10]:


encoder.fit_transform(df1['label'])


# In[11]:


df1['label'] = encoder.fit_transform(df1['label'])


# In[12]:


df1.head()


# In[13]:


# checking duplicates and 
df1.isnull().sum()


# In[14]:


df1.duplicated().sum()


# In[15]:


# removing duplicates
df1.drop_duplicates(keep='first',inplace=True)
df1


# In[16]:


df1.duplicated().sum()


# In[17]:


df1.shape


# # EDA

# In[18]:


df1.head()


# In[19]:


df1['label'].value_counts()


# In[20]:


import matplotlib.pyplot as plt
plt.pie(df1['label'].value_counts(), labels=['ham','spam'], autopct = "%0.2f")
plt.show()
# autopct parameter in pie() allows you to format the percentage labels that appear on the pie


# In[21]:


# Data is imbalanced


# In[22]:


import nltk


# In[23]:


nltk.download('punkt')


# In[24]:


df1['message'].apply(len)


# In[25]:


df1['num_characters']=df['message'].apply(len)


# In[26]:


df1


# In[27]:


# Number of words
df1['message'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[28]:


df1['num_words'] = df1['message'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[29]:


df1.head()


# In[30]:


df1['message'].apply(lambda x:(nltk.sent_tokenize(x)))


# In[31]:


df1['num_sentences'] = df1['message'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[32]:


df1[['num_words','num_characters','num_sentences']].describe()


# In[33]:


# ham
df1[df1['label']==0][['num_words','num_characters','num_sentences']].describe()


# In[34]:


# spam
df1[df1['label']==1][['num_words','num_characters','num_sentences']].describe()


# In[35]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[36]:


plt.figure(figsize=(12,6))
sns.histplot(df1[df1['label'] == 0]['num_characters'])
sns.histplot(df1[df1['label'] == 1]['num_characters'],color = 'red')


# In[37]:


plt.figure(figsize=(12,6))
sns.histplot(df1[df1['label'] == 0]['num_sentences'])
sns.histplot(df1[df1['label'] == 1]['num_sentences'],color = 'red')


# In[38]:


plt.figure(figsize=(12,6))
sns.histplot(df1[df1['label'] == 0]['num_words'])
sns.histplot(df1[df1['label'] == 1]['num_words'],color = 'red')

# spam start from 0 
# ham start from approx 25
# so not the big different
# no. of ham and spam ke base pe hm ni bta pa rhe hai ki spam ke words jada hai ya spam ke jada hai.


# In[39]:


# num_characters are more in Spam messages
# num_characters are less in Ham messages
sns.pairplot(df1,hue='label')


# In[40]:


df2 = df1.drop(columns=['message'])
df2


# In[41]:


df2.corr()


# In[42]:


plt.figure(figsize=(5,5))
sns.heatmap(df2.corr(),annot=True,square=True)


# # 3. Data Preprocessing

# In[43]:


from nltk.corpus import stopwords


# In[44]:


import string
string.punctuation


# In[45]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[46]:


def transform_text(text):
    text = text.lower()
    
    # Tokenize text
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:] 
    y.clear()   

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()   

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# In[47]:


nltk.download('stopwords')


# In[48]:


df1['message'].apply(transform_text)


# In[49]:


df1.head()


# In[50]:


df1['transformed_message']=df1['message'].apply(transform_text)


# In[51]:


df1.head()


# In[52]:


get_ipython().system('pip install wordcloud')


# In[53]:


from wordcloud import WordCloud


# In[54]:


wc=WordCloud(width=500,height=500,min_font_size=10,background_color='white')


# In[55]:


spam_wc=wc.generate(df1[df1['label']==1]['transformed_message'].str.cat(sep=" "))


# In[56]:


plt.imshow(spam_wc)


# In[57]:


wc=WordCloud(width=500,height=500,min_font_size=10,background_color='white')


# In[58]:


spam_wc=wc.generate(df1[df1['label']==0]['transformed_message'].str.cat(sep=" "))


# In[59]:


plt.imshow(spam_wc)


# In[60]:


df1[df1['label']==1]['transformed_message'].tolist()


# In[61]:


spam_corpus = []
for msg in df1[df1['label']==1]['transformed_message'].tolist():
    for word in msg.split():
        spam_corpus.append(word)


# In[62]:


len(spam_corpus)


# In[63]:


from collections import Counter
Counter(spam_corpus)


# In[64]:


# Adding in Dataframe
from collections import Counter
df3 = pd.DataFrame(Counter(spam_corpus).most_common(30))
df3


# In[65]:


df3 = df3.rename(columns={0:'Word',1:'Count'})
df3


# In[66]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.barplot(x='Word',y='Count',data = df3)
plt.xticks(rotation='vertical')
plt.show()


# In[ ]:




