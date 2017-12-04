
# coding: utf-8

# In[68]:

import pandas as pd
import numpy as np
from datetime import datetime
from gensim.models import Word2Vec
from nltk.tokenize import TweetTokenizer 
import gensim


from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer


# In[71]:

tokenizer = TweetTokenizer()
LabeledSentence = gensim.models.doc2vec.LabeledSentence 

def parse_dates(posix_time):
    return datetime.utcfromtimestamp(int(posix_time)).strftime('%Y-%m-%dT%H')

def tokenize(tweet):
    try:
        tokens = tokenizer.tokenize(tweet)
        return tokens
    except:
        return 'NC'
    
def postprocess(data):
    data['tokens'] = data['text'].map(tokenize) 
    data = data[data.tokens != 'NC']
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    return data

def buildWordVector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += tweet_w2v[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError: # handling the case where the token is not
                         # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec


# In[74]:

n_dim = 200
tweets = pd.read_csv('exported_tweets.csv')
prices = pd.read_csv('prices.csv')
tweets['created_at'] = tweets['created_at'].map(lambda x: str(x)[:-3])
tweets['timestamp'] = tweets['created_at'].apply(parse_dates)

data = tweets.merge(prices,on='timestamp',how='left')
data['tokens'] = data['text'].map(tokenize) 
#final_data = postprocess(data)



# In[75]:

x_train, x_test, y_train, y_test = train_test_split(np.array(data.tokens),
                                                    np.array(data.sentiment), test_size=0.2)


# In[76]:

tweet_w2v = Word2Vec(size=n_dim, min_count=10)
tweet_w2v.build_vocab(x_train)
tweet_w2v.train(x_train, total_examples=tweet_w2v.corpus_count, epochs=tweet_w2v.iter)


# In[77]:

vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
matrix = vectorizer.fit_transform(x_train)
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))


# In[79]:

from sklearn.preprocessing import scale
train_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in x_train])
train_vecs_w2v = scale(train_vecs_w2v)

test_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in x_test])
test_vecs_w2v = scale(test_vecs_w2v)


# In[85]:

#now we have training and testing datasets we can apply SVC, Guassian..etc
from sklearn.svm import SVC
svc_model = SVC(kernel='rbf', C=20, gamma=2)
svc_model.fit(train_vecs_w2v, y_train)
score = svc_model.score(test_vecs_w2v, y_test)
score


# In[ ]:



