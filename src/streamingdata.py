import configparser
import pandas as pd
import tweepy
from tweepy import OAuthHandler
import re
from cleanscopedata import clean_scope
import os

# read config

def connectTweepy():

    # # config parser 
    # config = configparser.ConfigParser()
    # config.read('config.ini')
    # consumer_key = config['twitter']['consumer_key']
    # consumer_secret = config['twitter']['consumer_secret']
    # access_token = config['twitter']['access_token']
    # access_token_secret = config['twitter']['access_token_secret']
    consumer_key = 'aFPmftKbB7FSHTaGxzynEb5FC'
    consumer_secret = 'pXLcyJFkxKWbQadcUbSM9kesP4rLZ8QXplzca64tKwsS4TnglN'
    access_token = '1275659737611284481-kqfUuvb4SZxWcy2IqJOgx7Hm7FGaW3'
    access_token_secret = '6XbkaVKJT8oOO0k6TmLrrnvpjOcCyX4XSmF38nX6lCH33'

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    return api

def collect_tweets(api,search_words,date_since):
    # Collect tweets
  tweets = tweepy.Cursor(api.search,
                q=search_words,
                lang="en"
                # since=date_since
               ).items(500)



  data = [[tweet.text] for tweet in tweets]
  tweet_text_df = pd.DataFrame(data=data, 
                      columns=['tweet_text'])
  return tweet_text_df


#if __name__ == "__main__":
      # search_words = input("Enter the search term: ")
  # print(search_words)
def fetchStreamingData(search_words,date_since):
    # search_words = '#climatechange'
    # date_since = "2018-11-16"
    api=connectTweepy()
    tweet_text_df = pd.DataFrame(collect_tweets(api,search_words,date_since))
    tweet_text_df.rename(columns={'tweet_text':'content'},inplace=True)
    write_data(tweet_text_df)
    #store raw data
    
 # tweet_text_df['tweet_text_cleaned'] = tweet_text_df.tweet_text.apply(clean_tweet)
 # tweet_text_df['tweet_sentiment'] = tweet_text_df.tweet_text_cleaned.apply(get_tweet_sentiment)
 

def write_data(X_scope):
    project_directory=os.getcwd()
    path=project_directory+"/data/raw/"
    isExist = os.path.exists(path)
    if not isExist:
      os.makedirs(path)
      print("The new directory is created!")
    X_scope.to_csv(path+"/raw_scope.csv" 
    # with open(path+'scope_features.pkl', "wb") as f:
    #   pkl.dump(X_predict, f)
   )
  
  
  
  
  
  
  
  
  
  
  
  
  

# # We create a tweet list as follows:
# tweets = api.user_timeline(screen_name="realDonaldTrump", count=200)
# print("Number of tweets extracted: {}.\n".format(len(tweets)))

# # We print the most recent 5 tweets:
# print("5 recent tweets:\n")
# for tweet in tweets[:5]:
#     print(tweet.text)
#     print()