# Import necessary modules
import os 
import re
import string
import pandas as pd
import numpy as np
import time
import pickle as pkl
from re import X
import sys
import scipy.sparse as sparse
import yaml
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
# Import necessary module
#from gensim.models import Word2Vec, KeyedVectors

import nltk
from sklearn.model_selection import train_test_split

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

params = yaml.safe_load(open("params.yaml"))["featurize"]

np.set_printoptions(suppress=True)

max_feature = params["max_features"]
ngrams = params["ngrams"]

  

def clean_text(text):
    text=text.lower()
    text = re.sub("@[A-Za-z0-9_]+","", text)
    text = re.sub("#[A-Za-z0-9_]+","", text)
    text=re.sub('\[.*?\]','',text)
    text=re.sub('[%s]'%re.escape(string.punctuation),'',text)
    text=re.sub('\w*\d\w*','',text)
    text=re.sub('\n','',text)
    text= re.sub(r'http\S+','', text)
    text=re.sub('[''"",,,]','',text)
    text = re.sub('[!@#$]', '', text)
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

def write_data(X_predict):
    print('while writing')
    print(X_predict.shape)
    project_directory=os.getcwd()
    path=project_directory+"/data/features/"
    isExist = os.path.exists(path)
    if not isExist:
      os.makedirs(path)
      print("The new directory is created!")
      
    with open(path+'scope_features.pkl', "wb") as f:
       pkl.dump(X_predict, f)
    #X_predict.to_csv(path+"/scope_features.csv",index=False)

  
  
def vectorize(X_predict):
    count_vectorizer = CountVectorizer(stop_words='english', strip_accents='ascii',
                                   min_df=.001,max_features=max_feature)
    
    X_predict_bow = count_vectorizer.fit_transform(X_predict['content'].values.astype('U'))
  
    # Convert matrices into a DataFrame
    X_predict_bow_df= pd.DataFrame(X_predict_bow.toarray())
  
    # Map the column names to vocabulary 
    X_predict_bow_df.columns = count_vectorizer.get_feature_names_out()
    # Drop 'content'
    X_predict.drop('content', axis=1, inplace=True)
    print('X_predict',X_predict.shape)
    print('X_predict',X_predict.head())
    # Merge data
    X_pred_vec = X_predict.reset_index().join(X_predict_bow_df)
    print('X_predict_bow_df',X_predict_bow_df.shape)
    print("X_pred_vec",X_pred_vec.shape)
    row,column=X_pred_vec.shape
    print(X_pred_vec.head())
    X_pred_vec_new=X_pred_vec.iloc[:, 1:column-1]
    print("X_pred_vec_new",X_pred_vec_new.shape)
    print(X_pred_vec_new.head())
    write_data(X_pred_vec_new)

def clean_scope(df):
  print('****** inside polarity********')
  df['content']=pd.DataFrame(df.content.apply(lambda x:clean_text(x)))
  
  # Drop rows with one or less characters in the tweet
  df.drop(df[df.content.str.len()<2].index, inplace=True)
  
  df['content']=df['content'].apply( lemma)
  df['content'] = df['content'].apply(lambda x: [item for item in 
                          x if item not in stop_word])

  #remove words with length less 2
  df.content=df.content.apply(lambda x: removewords(x))


  project_directory = os.getcwd() 
  print("current working directory", project_directory)
  print('project_directory',project_directory)
  path = project_directory+"/data/preprocessed"
  
  # Check whether the specified path exists or not
  isExist = os.path.exists(path)
  if not isExist:
    os.makedirs(path)
    print("The new directory is created for scope data")

  timestr = time.strftime("%Y%m%d-%H%M%S")
  print("d1 =", timestr)

  df.to_csv(path+"/scope_preocessed.csv")
  vectorize(df)
  
# projectdir=os.getcwd()
# path=projectdir+"/data/raw/"
# scope_filepath = os.path.join(path, "raw_scope.csv")
# tweet_text_df=pd.read_csv(scope_filepath,index_col=False)
# clean_scope(tweet_text_df)
    


    

    




        






























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