import os
import pickle
from re import X
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import yaml
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
# Import necessary modules
import pandas as pd
import numpy as np
#from gensim.models import Word2Vec, KeyedVectors
import nltk
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay, roc_auc_score,classification_report, accuracy_score
#pd.set_option('display.max_colwidth', 160)
import pickle as pkl

params = yaml.safe_load(open("params.yaml"))["featurize"]

np.set_printoptions(suppress=True)

max_feature = params["max_features"]
ngrams = params["ngrams"]



#save_matrix(df_test, test_words_tfidf_matrix, feature_names, test_output)

def read_data(path):
    
  df_origin = pd.read_csv(path,index_col=[0])
  df=df_origin.copy()
  # Change sentiment of the tweets with only mentions to "neutral"
  print('******Read Data********')
  return df

def write_data(X_train,X_test,y_train,y_test):
    project_directory=os.getcwd()
    path=project_directory+"/data/features/"
    train_data=X_train.merge(y_train, on="tweet_id",how='inner')
    test_data=X_test.merge(y_test, on="tweet_id",how='inner')
 
   
    isExist = os.path.exists(path)
    if not isExist:
      os.makedirs(path)
      print("The new directory is created!")
      
    with open(path+'train.pkl', "wb") as f:
     pkl.dump(train_data, f)
      
    #to save it
    with open(path+'test.pkl', "wb") as f:
      pkl.dump(test_data, f)
    print('******write Data********')
      
def split(df):
  # Create features and target
  y = df[['tweet_id','sentiment', 'polarity']]
  X = df.drop(columns=['sentiment','polarity'], axis=1)
  
  # Split training and testing data
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=57)
  print('******split Data********')
  return X_train,X_test,y_train,y_test

  
  

if __name__ == "__main__":
  project_directory = os.getcwd() 
  print("current working directory", project_directory)
  path=project_directory+"/data/preprocessed/prepared.csv"
  df=read_data(path)
  X_train,X_test,y_train,y_test=split(df)
  #Vectorize 
 # X_train_vec, X_test_vec=get_TFIDFVectors(X_train,X_test)
 # BAG_Words 
  # Initialize count vectorizer
  count_vectorizer = CountVectorizer(stop_words='english', strip_accents='ascii',
                                   min_df=.001,max_features=max_feature)
  # Create count train and test variables
  X_train_bow = count_vectorizer.fit_transform(X_train['content'].values.astype('U'))
  X_test_bow = count_vectorizer.transform(X_test['content'].values.astype('U'))

  # Convert matrices into a DataFrame
  X_train_bow_df = pd.DataFrame(X_train_bow.toarray())
  X_test_bow_df = pd.DataFrame(X_test_bow.toarray())

  # Map the column names to vocabulary 
  X_train_bow_df.columns = count_vectorizer.get_feature_names_out()
  X_test_bow_df.columns = count_vectorizer.get_feature_names_out()

  # Drop 'content'
  X_train.drop('content', axis=1, inplace=True)
  X_test.drop('content', axis=1, inplace=True)
  print('X_train',X_train.shape)
  print('X_train_bow_df',X_train_bow_df.shape)

  

  # Merge data
  X_train_vec = X_train.reset_index().join(X_train_bow_df)
  X_test_vec = X_test.reset_index().join(X_test_bow_df)
  X_train_vec.drop(columns='index',inplace=True)
  X_test_vec.drop(columns='index',inplace=True)
  print('X_train_vec',X_train_vec.shape)
  print('X_train_bow_df',X_train_bow_df.head())
  print(X_train_vec.head())
   
  
  write_data(X_train_vec,X_test_vec,y_train,y_test)
    
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
# def get_TFIDFVectors(X_train,X_test):
#   # Initialize tfidf vectorizer
#   tfidf_vectorizer = TfidfVectorizer(stop_words='english', min_df=100,max_features=max_feature)

#   print('****** vectorize********')

#   # Create tfidf train and test variables
#   X_train_tfidf = tfidf_vectorizer.fit_transform(df['content'].values.astype('U'))
#   X_test_tfidf = tfidf_vectorizer.transform(df['content'].values.astype('U'))  


#   # Map the column names to vocabulary 
#   X_train_tfidf.columns = tfidf_vectorizer.get_feature_names_out()
#   X_test_tfidf.columns = tfidf_vectorizer.get_feature_names_out()

#   # Drop 'content'
#   X_train.drop('content', axis=1, inplace=True)
#   X_test.drop('content', axis=1, inplace=True)
#   print('****** vectorize********')
#   # # Merge data
#   # X_train_vec = X_train.join(X_train_tfidf)
#   # X_test_vec = X_test.join(X_test_tfidf)
 
#   return X_train_vec, X_test_vec
  
 
 
# if len(sys.argv) != 3 and len(sys.argv) != 5:
#     sys.stderr.write("Arguments error. Usage:\n")
#     sys.stderr.write("\tpython featurization.py data-dir-path features-dir-path\n")
#     sys.exit(1)

# train_input = os.path.join(sys.argv[1], "train.tsv")
# test_input = os.path.join(sys.argv[1], "test.tsv")
# train_output = os.path.join(sys.argv[2], "train.pkl")
# test_output = os.path.join(sys.argv[2], "test.pkl")
