import sys
import tweepy
import csv

#pass security information to variables
consumer_key="EwUtIn4Oj9hef3gISco6gS1bm"
consumer_secret="OZEQ7iVEk2vkGwaWigQ9tMkJXW8hWKqBjMZfiNhMdM02R8XqXa"
access_key = "2157294756-Gcw3huSBAT86IRCj6zmJo7bfm6C0afMZqkXSd89"
access_secret = "7fPZRTfHJ5SPCm41Y0FdC3b30KNTKGnoUxD53Xs4cf2Je"


#use variables to access twitter
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)


#create an object called 'customStreamListener'


class CustomStreamListener(tweepy.StreamListener):

    def __init__(self):
        self.f = open('bitcoin-tweets.txt', 'w')
        self.f.write('Author,Date,Text\n')
        self.writer = csv.writer(self.f, lineterminator='\n')
        self.api = tweepy.API(auth)

    def on_status(self, status):
        print([status.author.screen_name, status.created_at, status.text])
        self.writer.writerow([status.author.screen_name.replace('\n', ' ').replace('\r', ''), status.created_at, status.text.replace('\n', ' ').replace('\r', '')])


    def on_error(self, status_code):
        print(sys.stderr, 'Encountered error with status code:', status_code)
        self.f.close()

        return False # Don't kill the stream

    def on_timeout(self):
        print(sys.stderr, 'Timeout...')
        return True # Don't kill the stream


streamingAPI = tweepy.streaming.Stream(auth, CustomStreamListener())
streamingAPI.filter(track=['bitcoin'])