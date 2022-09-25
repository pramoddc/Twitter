# Import necessary modules
import os 
import re
import string
import pandas as pd
import numpy as np

#important libraries for preprocessing using NLTK
import nltk
nltk.download
nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('stopwords')

from nltk import word_tokenize, FreqDist
from nltk.corpus import stopwords
stop_word = set(stopwords.words('english'))
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
  
# mongodb connection  
import pymongo
from pymongo import MongoClient


def read_data():
  client = pymongo.MongoClient("mongodb://sampada:sampada@ac-kercfkw-shard-00-00.ihjo4b6.mongodb.net:27017,ac-kercfkw-shard-00-01.ihjo4b6.mongodb.net:27017,ac-kercfkw-shard-00-02.ihjo4b6.mongodb.net:27017/?ssl=true&replicaSet=atlas-6y3gj6-shard-0&authSource=admin&retryWrites=true&w=majority")
  mydatabase = client['Twitter']
  mycollection = mydatabase['tweets']
  print("MongoDB Collection Connection Successful")
  mongo_docs =(mycollection.find())
  df_origin = pd.DataFrame(list(mongo_docs))
  #df_origin = pd.read_csv(path)
  df=df_origin.copy()
  # Change sentiment of the tweets with only mentions to "neutral"
  df.loc[df.content.str.replace("@[^\s]+", "").str.len()<3, 'sentiment'] = "neutral"
  print('******Read Data********')
  return df


def clean_text(text):
    text=text.lower()
    text = re.sub("@[A-Za-z0-9_]+","", text)
  #  text = re.sub("#[A-Za-z0-9_]+","", text)
    text=re.sub('\[.*?\]','',text)
    text=re.sub('[%s]'%re.escape(string.punctuation),'',text)
    text=re.sub('\w*\d\w*','',text)
    text=re.sub('\n','',text)
    text= re.sub(r'http\S+','', text)
    text=re.sub('[''"",,,]','',text)
    return text


def lemma(text):
      lemmatizer = nltk.stem.WordNetLemmatizer()
      w_tokenizer = TweetTokenizer()
      return [(lemmatizer.lemmatize(w)) for w in 
                                     w_tokenizer.tokenize((text))]

def removewords(text):
  final_text=""
  for word in text:
    if len(word)>2:
      final_text=final_text+" "+word
  return final_text


def main(df):
  print('******Inside main function********')
  df['content']=pd.DataFrame(df.content.apply(lambda x:clean_text(x)))
  # Drop rows with sentiment "empty"
  df = df[df.sentiment != 'empty']

  # Drop rows with one or less characters in the tweet
  df.drop(df[df.content.str.len()<2].index, inplace=True)

  print('******Step 3********')
  # Create a sentiment dictionary to map EMOTIONS to SENTIMENTS.
  sentiment_dict = {'boredom': 'negative',
                    'hate': 'negative',
                    'sadness': 'negative',
                    'anger': 'negative',
                    'worry': 'negative',
                    'relief': 'positive',
                  # 'empty': 'neutral',
                    'happiness': 'positive',
                    'love': 'positive',
                    'enthusiasm': 'positive',
                    'neutral': 'neutral',
                    'surprise':'positive',
                    'fun': 'positive'
                  }
  # Create a feature "polarity"
  df['polarity'] = df.sentiment.map(sentiment_dict)
  
  print('******After setting Polarity********')
  
  df['content']=df['content'].apply( lemma)
  df['content'] = df['content'].apply(lambda x: [item for item in 
                          x if item not in stop_word])

  # Drop unnecessary columns
  df.drop(columns=['author'], axis=1, inplace=True)

  #remove words with length less 2
  df.content=df.content.apply(lambda x: removewords(x))
  #df.drop(columns=['tweet_id'],inplace=True,axis=1)
  df=df[['tweet_id','content','polarity','sentiment']]
  print(df.head())
  print('project_directory',project_directory)
  path = project_directory+"/data/preprocessed"
 # Check whether the specified path exists or not
  isExist = os.path.exists(path)
  if not isExist:
    os.makedirs(path)
    print("The new directory is created!")
  df.to_csv(path+"/prepared.csv")
  
if __name__=="__main__":
  project_directory = os.getcwd() 
  print("current working directory", project_directory)

  #path=project_directory+"/data/raw/text_emotion.csv"
  df=read_data()
  main(df)
 


      






























'''import io
import os
import random
import re
import sys
import xml.etree.ElementTree

import yaml

 params = yaml.safe_load(open("params.yaml"))["prepare"]

if len(sys.argv) != 2:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython prepare.py data-file\n")
    sys.exit(1)

# Test data set split ratio
split = params["split"]
random.seed(params["seed"])

input = sys.argv[1]
output_train = os.path.join("data", "prepared", "train.tsv")
output_test = os.path.join("data", "prepared", "test.tsv")


def process_posts(fd_in, fd_out_train, fd_out_test, target_tag):
    num = 1
    for line in fd_in:
        try:
            fd_out = fd_out_train if random.random() > split else fd_out_test
            attr = xml.etree.ElementTree.fromstring(line).attrib

            pid = attr.get("Id", "")
            label = 1 if target_tag in attr.get("Tags", "") else 0
            title = re.sub(r"\s+", " ", attr.get("Title", "")).strip()
            body = re.sub(r"\s+", " ", attr.get("Body", "")).strip()
            text = title + " " + body

            fd_out.write("{}\t{}\t{}\n".format(pid, label, text))

            num += 1
        except Exception as ex:
            sys.stderr.write(f"Skipping the broken line {num}: {ex}\n")


os.makedirs(os.path.join("data", "prepared"), exist_ok=True)

with io.open(input, encoding="utf8") as fd_in:
    with io.open(output_train, "w", encoding="utf8") as fd_out_train:
        with io.open(output_test, "w", encoding="utf8") as fd_out_test:
            process_posts(fd_in, fd_out_train, fd_out_test, "<r>")
 '''