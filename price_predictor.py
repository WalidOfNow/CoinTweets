
# coding: utf-8

# In[48]:

import pandas as pd
import numpy as np
from datetime import datetime
from gensim.models import Word2Vec
from nltk.tokenize import TweetTokenizer 
import gensim
import enchant



from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer


import matplotlib.pyplot as plt

# In[49]:

tokenizer = TweetTokenizer()
LabeledSentence = gensim.models.doc2vec.LabeledSentence
d = enchant.Dict("en_US")

def parse_dates(posix_time):
    return datetime.utcfromtimestamp(int(posix_time)).strftime('%Y-%m-%dT%H')

def tokenize(tweet):
    try:
        tokens = tokenizer.tokenize(tweet)
        for word in tokens:
            if not d.check(word):
                tokens.remove(word)

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

print("reading data and building dataframes...")
n_dim = 200
trained_tweets = pd.read_csv('exported_tweets.csv')
training_data = pd.read_csv('bitcoin-train-sentiments.csv')
non_trained_tweets = pd.read_csv('exported_tweets_new.csv')

prices = pd.read_csv('prices.csv')
non_trained_tweets['created_at'] = non_trained_tweets['created_at'].map(lambda x: str(x)[:-3])
non_trained_tweets['timestamp'] = non_trained_tweets['created_at'].apply(parse_dates)

data = non_trained_tweets.merge(prices,on='timestamp',how='left')
#data['tokens'] = data['text'].map(tokenize) 
#final_data = postprocess(data)
training_data = training_data.append(trained_tweets)
training_data['tokens'] = training_data['text'].map(tokenize) 
data['tokens'] = data['text'].map(tokenize)

#training_data


# In[60]:

#training_data


# In[89]:

#x_train, x_test, y_train, y_test = train_test_split(np.array(training_data.tokens),
#                                                    np.array(training_data.sentiment), test_size=0.2)


x_train = training_data.tokens
y_train = training_data.sentiment

x_test = data.tokens


# In[90]:

print("training word2vec model...")

tweet_w2v = Word2Vec(size=n_dim, min_count=10)
tweet_w2v.build_vocab(x_train)
tweet_w2v.train(x_train, total_examples=tweet_w2v.corpus_count, epochs=tweet_w2v.iter)


print("fitting data to vectorizer...")

vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
matrix = vectorizer.fit_transform(x_train)
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))


print("building word vectors for model...")

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
#
# model.fit(train_vecs_w2v, y_train, epochs=9, batch_size=32, verbose=2)




#now we have training and testing datasets we can apply SVC, Guassian..etc

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

#model = SVC(kernel='linear', C=200)

#model = KNeighborsClassifier(n_neighbors = 100)
#model = GaussianNB()


#print("fitting data to svc model...")

#model.fit(train_vecs_w2v, y_train)
#sentiment_predictions = model.predict(test_vecs_w2v)



# In[107]:

#data['sentiment'] = sentiment_predictions

#predicted_sentiments = data
#print(data[data['sentiment'] == -1])

predicted_sentiments = pd.read_csv('not_the_worst_predicted_sentiments.csv')

#predicted_sentiments['created_at'] = pd.to_datetime(predicted_sentiments['created_at'], unit='s')
#print(predicted_sentiments)

from scipy import stats

#https://stackoverflow.com/questions/15222754/group-by-pandas-dataframe-and-select-most-common-string-factor
group = predicted_sentiments.groupby('timestamp')

ts = group['timestamp']

sentiments = group['sentiment'].agg(lambda x: stats.mode(x)[0][0])
price = group['price'].mean()
vol = group['volume'].mean()
num_tweets = group['text'].count()


final_df = pd.DataFrame(data={'sentiment':sentiments, 'price':price, 'volume':vol, 'num_tweets':num_tweets})

final_df = final_df.reset_index()


fig, ax = plt.subplots()
#ax.plot(final_df['timestamp'], final_df['price'])
def color_it(sent):
    if sent == 1:
        return 'G'
    elif sent==0:
        return 'B'
    else:
        return 'R'



def get_price_change(data):
    #data 2 = curr price
    #data 5 = next price
    if  abs(data[5]-data[2]) <= (data[2] * 0.005):
        return 0
    elif data[5]-data[2] > 0:
        return 1
    else:
        return -1


final_df['price_next'] = final_df['price'].shift(-1)
final_df = final_df.dropna()
final_df['price-change'] = final_df.apply(get_price_change, axis=1)

final_df['timestamp'] = pd.to_datetime(final_df['timestamp']).values.astype(np.int64)
final_df['volume'] = final_df['volume'].values.astype(np.int64)
final_df['price_next'] = final_df['price_next'].values.astype(np.int64)

print(final_df)

x2_train, x2_test, y2_train, y2_test = train_test_split(final_df[['timestamp', 'sentiment', 'volume', 'num_tweets']],
                                                    final_df['price-change'])

#model2 = SVC(kernel='linear', C=0.01)
model2 = KNeighborsClassifier(n_neighbors = 100)
#model2 = GaussianNB()
model2.fit(x2_train, y2_train)
print(model2.score(x2_test, y2_test))




#ax.scatter(final_df['timestamp'], final_df['price'], c=final_df['sentiment'].apply(color_it), s=3)
# start, end = ax.get_xlim()
# ax.xaxis.set_ticks(np.arange(start, end, 100))
# plt.xticks(rotation=40)
# plt.show()





# In[108]:

#data.to_csv('not_the_worst_predicted_sentiments.csv', index=False)


# In[ ]:

