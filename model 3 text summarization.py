#!/usr/bin/env python
# coding: utf-8

# In[1]:


import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation


# In[2]:


nlp = spacy.load('en_core_web_sm')


# In[3]:


text = """ Maria Sharapova has basically no friends as tennis players on the WTA Tour. The Russian player has no problems in openly speaking about it and in a recent interview she said: ‘I don’t really hide any feelings too much.
I think everyone knows this is my job here. When I’m on the courts or when I’m on the court playing, I’m a competitor and I want to beat every single person whether they’re in the locker room or across the net.
So I’m not the one to strike up a conversation about the weather and know that in the next few minutes I have to go and try to win a tennis match.
I’m a pretty competitive girl. I say my hellos, but I’m not sending any players flowers as well. Uhm, I’m not really friendly or close to many players.
I have not a lot of friends away from the courts.’ When she said she is not really close to a lot of players, is that something strategic that she is doing? Is it different on the men’s tour than the women’s tour? ‘No, not at all.
I think just because you’re in the same sport doesn’t mean that you have to be friends with everyone just because you’re categorized, you’re a tennis player, so you’re going to get along with tennis players.
I think every person has different interests. I have friends that have completely different jobs and interests, and I’ve met them in very different parts of my life.
I think everyone just thinks because we’re tennis players we should be the greatest of friends. But ultimately tennis is just a very small part of what we do.
There are so many other things that we’re interested in, that we do. """


# In[4]:


stopword = list(STOP_WORDS)


# In[5]:


doc = nlp(text) 


# In[6]:


tokens = [token.text for token in doc]
print(tokens)


# In[7]:


punctuation
# list of punctuation


# # Text Cleaning 
# 

# In[8]:


word_frequencies = {}

for word in doc:
    if word.text.lower() not in stopword:
        if word.text.lower() not in punctuation:
            if word.text not in word_frequencies.keys():
                word_frequencies[word.text] = 1
            else:
                word_frequencies[word.text] +=1


# In[9]:


print(word_frequencies)


# In[10]:


max_frequencies = max(word_frequencies.values())


# In[11]:


max_frequencies


# In[12]:


for word in word_frequencies.keys():
    word_frequencies[word] = word_frequencies[word] / max_frequencies


# In[13]:


print(word_frequencies)


# # Sentence Tokenization

# In[14]:


sentence_tokens = [sent for sent in doc.sents]
print(sentence_tokens)


# In[15]:


len(sentence_tokens)


# In[16]:


sentence_score = {}

for sent in sentence_tokens:
    for word in sent:
        if word.text.lower() in word_frequencies.keys():
            if sent not in sentence_score.keys():
                sentence_score[sent] = word_frequencies[word.text.lower()]
            else:
                sentence_score[sent] += word_frequencies[word.text.lower()]


# In[17]:


print(sentence_score)


# In[18]:


18*(30/100)


# In[19]:


200*(20/100)


# In[20]:


from heapq import nlargest


# In[21]:


select_length = int(len(sentence_tokens)) * 0.3


# In[22]:


print(select_length)


# # Getting the Summary

# In[23]:


summary = nlargest(n=int(select_length), iterable=sentence_score, key = sentence_score.get)


# In[24]:


print(summary)


# In[25]:


final_summary = [word.text for word in summary]


# In[26]:


final_summary


# In[27]:


len(text)


# In[28]:


len(summary) 


# In[ ]:




