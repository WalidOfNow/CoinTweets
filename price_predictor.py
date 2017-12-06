import pandas as pd
import numpy as np
from datetime import datetime
from gensim.models import Word2Vec
from nltk.tokenize import TweetTokenizer 
import gensim


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import scale
from sklearn.naive_bayes import GaussianNB

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


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


print("reading data and building dataframes...")
n_dim = 200
trained_tweets = pd.read_csv('exported_tweets.csv')
training_data = pd.read_csv('bitcoin-train-sentiments.csv')
non_trained_tweets = pd.read_csv('exported_tweets_new.csv')

prices = pd.read_csv('prices.csv')
non_trained_tweets['created_at'] = non_trained_tweets['created_at'].map(lambda x: str(x)[:-3])
non_trained_tweets['timestamp'] = non_trained_tweets['created_at'].apply(parse_dates)

data = non_trained_tweets.merge(prices,on='timestamp',how='left')

training_data = training_data.append(trained_tweets)
training_data['tokens'] = training_data['text'].map(tokenize) 
data['tokens'] = data['text'].map(tokenize)


# split our data
x_train = training_data.tokens
y_train = training_data.sentiment

x_test = data.tokens



print("training word2vec model...")

tweet_w2v = Word2Vec(size=n_dim, min_count=10)
tweet_w2v.build_vocab(x_train)
tweet_w2v.train(x_train, total_examples=tweet_w2v.corpus_count, epochs=tweet_w2v.iter)


print("fitting data to vectorizer...")

vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
matrix = vectorizer.fit_transform(x_train)
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))


print("building word vectors for model...")

train_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in x_train])
train_vecs_w2v = scale(train_vecs_w2v)

test_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in x_test])
test_vecs_w2v = scale(test_vecs_w2v)


#now we have training and testing datasets we can apply SVC, Guassian..etc

model = SVC(kernel='linear', C=100)
#model = GaussianNB()

print("fitting data to svc model (can take up to a few minutes)...")

model.fit(train_vecs_w2v, y_train)
sentiment_predictions = model.predict(test_vecs_w2v)
data['sentiment'] = sentiment_predictions
predicted_sentiments = data

print("aggregating data and generating graph(s)...")


from scipy import stats

group = predicted_sentiments.groupby('timestamp')

#https://stackoverflow.com/questions/15222754/group-by-pandas-dataframe-and-select-most-common-string-factor
sentiment = group['sentiment'].agg(lambda x: stats.mode(x)[0][0])
price = group['price'].mean()
vol = group['volume'].mean()
num_tweets = group['text'].count()
sentiment_sum = group['sentiment'].sum()

final_df = pd.DataFrame(data={'price':price, 'sentiment': sentiment, 'volume':vol, 'num_tweets':num_tweets, 'sentiment_sum': sentiment_sum})

final_df = final_df.reset_index()

def color_it(sent):
    if sent == 1:
        return 'G'
    elif sent==0:
        return 'B'
    else:
        return 'R'



def get_price_change(data):

    # if the price change was within 0.1% of the original price, then we treat it as if no price change happened
    if abs(data['price_next']-data['price']) <= data['price']*0.001:
        return 0
    if data['price_next']-data['price'] > 0:
        return 1
    else:
        return -1

#
final_df['price_next'] = final_df['price'].shift(-1)
final_df = final_df.dropna()
final_df['price-change'] = final_df.apply(get_price_change, axis=1)

final_df['correct'] = (final_df['sentiment'] == final_df['price-change'])

print("total accuracy of sentiment correlating to market price: ", len(final_df[final_df['correct'] == True]) / len(final_df['correct']))

final_df['volume'] = final_df['volume'].values.astype(np.int64)
final_df['price_next'] = final_df['price_next'].values.astype(np.int64)


#fig, ax = plt.subplots()
plt.scatter(final_df['timestamp'], final_df['price'], c=final_df['sentiment'].apply(color_it), marker='.')

#https://stackoverflow.com/questions/41598935/choose-matplotlib-xticks-frequency
cx = np.arange(final_df['timestamp'].shape[0])
my_date_ticks = np.array(final_df['timestamp'])
freq = 365
plt.xticks(cx[::freq], my_date_ticks[::freq], rotation=40)

plt.xlabel("date")
plt.ylabel("price in USD")


red_patch = mpatches.Patch(color='red', label='negative sentiment')
green_patch = mpatches.Patch(color='green', label='positive sentiment')
blue_patch = mpatches.Patch(color='blue', label='neutral sentiment')


plt.legend(handles=[red_patch, green_patch, blue_patch])
plt.title("twitter sentiment bitcoin market price per hour")
plt.show()

# This generates our number of tweets per hour graph
# But it is very slow! Uncomment at your will!

# plt.bar(final_df['timestamp'], final_df['num_tweets'])
# plt.title("number of tweets per hour")
# plt.xlabel('date')
# plt.ylabel('number of tweets')
# plt.xticks(cx[::freq], my_date_ticks[::freq], rotation=40)
# plt.show()
