import os
import pickle
import yaml
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import pickle as pkl
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay, roc_auc_score,classification_report, accuracy_score

params = yaml.safe_load(open("params.yaml"))["train"]


def read_data(project_directory):
  
  print("current working directory", project_directory)
  path=project_directory+"/data/features/train.pkl"

  #to load it
  with open(path, "rb") as f:
        train_df= pkl.load(f)
 
  # Change sentiment of the tweets with only mentions to "neutral"
  print('******Read Data********')
  return train_df

def build_model_polarity(train_df, project_directory):
    print('******Build model********')
    # Create a MulitnomialNB model
    clf_nb = MultinomialNB()
    print(train_df.columns)
    train_df.dropna(inplace=True)
    y_train=train_df['polarity']
    X_train=train_df.drop(columns=['polarity','sentiment','tweet_id'],axis=1) 
    clf_nb.fit(X_train, y_train)

    #save model 
    outputpath=project_directory
    isExist = os.path.exists(outputpath)
    if not isExist:
      os.makedirs(outputpath)
      print("The new directory is created for polarity")
    with open(outputpath+"/polarityclassifier.pkl", "wb") as fd:
      pickle.dump(clf_nb, fd)
      
def build_model_sentiment(train_df, project_directory):
    print('******Build model********')
    # Create a MulitnomialNB model
    clf_nb = MultinomialNB()
    print(train_df.columns)
    train_df.dropna(inplace=True)
    print('***************************')
    print(train_df.head())
    y_train=train_df['sentiment']
    X_train=train_df.drop(columns=['polarity','sentiment','tweet_id'],axis=1) 
    print("X_train",X_train)
    clf_nb.fit(X_train, y_train)
    
    #save model 
    outputpath=project_directory
    isExist = os.path.exists(outputpath)
    if not isExist:
      os.makedirs(outputpath)
      print("The new directory is created for sentiment")
    
    with open(outputpath+"/sentimentclassifier.pkl", "wb") as fd:
      pickle.dump(clf_nb, fd)


if __name__=="__main__":
  project_directory = os.getcwd() 
  train_df=read_data(project_directory)
  build_model_sentiment(train_df,project_directory)
  build_model_polarity(train_df,project_directory)
  
 

''' if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython train.py features model\n")
    sys.exit(1)

input = sys.argv[1]
output = sys.argv[2]
seed = params["seed"]
n_est = params["n_est"]
min_split = params["min_split"]

with open(os.path.join(input, "train.pkl"), "rb") as fd:
    matrix, _ = pickle.load(fd)

labels = np.squeeze(matrix[:, 1].toarray())
x = matrix[:, 2:]

sys.stderr.write("Input matrix size {}\n".format(matrix.shape))
sys.stderr.write("X matrix size {}\n".format(x.shape))
sys.stderr.write("Y matrix size {}\n".format(labels.shape))

clf = RandomForestClassifier(
    n_estimators=n_est, min_samples_split=min_split, n_jobs=2, random_state=seed
)

'''
 


