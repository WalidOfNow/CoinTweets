
# coding: utf-8

# In[48]:

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


# In[49]:

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


# In[59]:

n_dim = 200
trained_tweets = pd.read_csv('exported_tweets.csv')
training_data = pd.read_csv('bitcoin-train-sentiments.csv')
non_trained_tweets = pd.read_csv('non_trained_tweets.csv')

prices = pd.read_csv('prices.csv')
non_trained_tweets['created_at'] = non_trained_tweets['created_at'].map(lambda x: str(x)[:-3])
non_trained_tweets['timestamp'] = non_trained_tweets['created_at'].apply(parse_dates)

data = non_trained_tweets.merge(prices,on='timestamp',how='left')
#data['tokens'] = data['text'].map(tokenize) 
#final_data = postprocess(data)
training_data = training_data.append(tweets)
training_data['tokens'] = training_data['text'].map(tokenize) 
data['tokens'] = data['text'].map(tokenize)

training_data


# In[60]:

training_data


# In[89]:

#x_train, x_test, y_train, y_test = train_test_split(np.array(training_data.tokens),
#                                                    np.array(training_data.sentiment), test_size=0.2)
x_train = training_data.tokens
y_train = training_data.sentiment

x_test = data.tokens


# In[90]:

tweet_w2v = Word2Vec(size=n_dim, min_count=10)
tweet_w2v.build_vocab(x_train)
tweet_w2v.train(x_train, total_examples=tweet_w2v.corpus_count, epochs=tweet_w2v.iter)


# In[91]:

vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
matrix = vectorizer.fit_transform(x_train)
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))


# In[94]:

from sklearn.preprocessing import scale
train_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in x_train])
train_vecs_w2v = scale(train_vecs_w2v)

test_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in x_test])
test_vecs_w2v = scale(test_vecs_w2v)


# In[106]:

# model = Sequential()
# model.add(Dense(32, activation='relu', input_dim=200))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(optimizer='rmsprop',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

# model.fit(train_vecs_w2v, y_train, epochs=9, batch_size=32, verbose=2)

#now we have training and testing datasets we can apply SVC, Guassian..etc
from sklearn.svm import SVC
svc_model = SVC(kernel='linear', C=200)
svc_model.fit(train_vecs_w2v, y_train)
sentiment_predictions = svc_model.predict(test_vecs_w2v)





# In[107]:

data['sentiment'] = sentiment_predictions
data[data['sentiment'] == -1]


# In[108]:

data.to_csv('predicted_sentiments.csv', index=False)


# In[ ]:



