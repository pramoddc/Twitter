import json
import math
import os
import pickle as pkl

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import tree
from dvclive import Live
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.metrics import auc,confusion_matrix
from sklearn.preprocessing import label_binarize
from os.path import exists



def read_data(project_directory):     
  print("reading data")
  path=project_directory+"/data/features/test.pkl"
  scores = {} # scores is an empty dict already
  #to load it
  with open(path, "rb") as f:
        test_df= pkl.load(f)
  return test_df

def load_model(project_directory,modelname):
    print("loading model")
    path=project_directory+"/"+modelname
    #to load it
    with open(path, "rb") as f:
           model= pkl.load(f)
    return model
       
def calculate_tpr_fpr(y_real, y_pred):
    # Calculates the confusion matrix and recover each element
    cm = confusion_matrix(y_real, y_pred)
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]   
    # Calculates tpr and fpr
    tpr =  TP/(TP + FN) # sensitivity - true positive rate
    fpr = 1 - TN/(TN+FP) # 1-specificity - false positive rate
    return tpr, fpr    

def get_all_roc_coordinates(y_real, y_proba):
    tpr_list = [0]
    fpr_list = [0]
    for i in range(len(y_proba)):
        threshold = y_proba[i]
        y_pred = (y_proba >= threshold)
        tpr, fpr = calculate_tpr_fpr(y_real, y_pred)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    return tpr_list, fpr_list

def evaluate_model (project_directory,test_df,clf_nb,targetcolumnname):   
    test_df.dropna(inplace=True)
    print("")
    #Creating X and Y 
    y_test=pd.DataFrame()
    ROC_Curve_name=""
    precission_recall_name=""
    if targetcolumnname=="polarity":
        y_test=test_df['polarity']
        ROC_Curve_name="polarity_roc_curve.png"
        precission_recall_name="polarity_precision_recall.json"
    elif targetcolumnname=="sentiment":
        y_test=test_df['sentiment']
        ROC_Curve_name="sentiment_roc_curve.png"
        precission_recall_name="sentiment_precision_recall.json"
    X_test=test_df.drop(columns=['polarity','sentiment','tweet_id'],axis=1) 
   
    #Get predictions
    y_pred_proba = clf_nb.predict_proba(X_test)      
    
    #Classes
    classes=np.unique(y_test)
    n_classes=classes.shape[0]
    
    #Binarizing labels
    y_test_bin=label_binarize(y_test,classes=classes)
    
    #find FPR and TPR for all the classes
    tpr=dict()
    fpr=dict()
    roc_auc=dict()
    threshold=dict()
    #tpr_list, fpr_list=get_all_roc_coordinates(y_test_bin,y_pred_proba)
    for i in range(n_classes):
        fpr[i],tpr[i],threshold[i]=roc_curve(y_test_bin[:,i],y_pred_proba[:,i])
        roc_auc[i]=auc(fpr[i],tpr[i])

    # Use dvclive to log a few simple plots ...
    eval_path=project_directory+"/dvc_plots"
    live = Live(eval_path)
    #ploting ROC AUC 
    plt.figure()
    fig, axes = plt.subplots(dpi=100)
    fig.subplots_adjust(bottom=0.2, top=0.95)
    lw = 2
    for i in range(n_classes):
        plt.plot(
            fpr[i],
            tpr[i],
            lw=lw,
            label='%s ROC curve (area = %0.2f)'%(classes[i],roc_auc[i]))
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver operating characteristic")
        plt.legend(loc="lower right")
    #path=project_directory+"/evaluation/test/plots/"+ROC_Curve_name
    path=project_directory+"/evaluation/test/plots/"

    isExist = os.path.exists(path)
    if not isExist:
      os.makedirs(path)
      print("The directory is not present! Creating the directory now")
    else:
        print("The directory is present!")
   
    filepath = os.path.join(path, ROC_Curve_name)
    file_exists = exists(filepath)
    print("file_exists before",file_exists)
    
    print('filepath', filepath)
    if not file_exists:
        with open(filepath,"x") as emptyfile:
          pass  
    file_exists = exists(filepath)          
    print("file_exists after",file_exists)
  
    print('path', path)
    fig.savefig(filepath)
        
        
    #saving Precission and Recall as Json 
    nth_point = math.ceil(len(threshold) / 1000)
    prc_points = list(zip(fpr, tpr, threshold))[::nth_point]
    
    # eval_path=project_directory+"\\evaluation\\test\\"
    eval_path=project_directory+"/evaluation/test/"

    FolderExist = os.path.exists(eval_path)
    print('Filepath : ',FolderExist)

    prc_file = os.path.join(eval_path, "plots", precission_recall_name)

    isExist = os.path.exists(prc_file)
    print('Filepath exists : ',prc_file)
    if not isExist:
        with open(prc_file,"x") as emptyfile:
          pass  
    file_exists = exists(prc_file)          
    print("file_exists after",file_exists)
    
     # Check whether the specified path exists or not
    with open(prc_file, "w") as fd:
        json.dump(
            {
                "prc": [
                    {"precision": p, "recall": r, "threshold": t}
                    for p, r, t in prc_points
                ]
            },
            fd,
            indent=4,
        )


if __name__=="__main__":
  project_directory = os.getcwd() 
  test_df=read_data(project_directory)
  #Polarity model
  modelname="polarityclassifier.pkl"
  polarity_model=load_model(project_directory,modelname)
  evaluate_model(project_directory,test_df,polarity_model,"polarity")
  
  #sentiment classifier
  modelname="sentimentclassifier.pkl"
  sentiment_model=load_model(project_directory,modelname)
  evaluate_model(project_directory,test_df,sentiment_model,"sentiment")
  
'''   
  

if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython evaluate.py model features\n")
    sys.exit(1)

model_file = sys.argv[1]
train_file = os.path.join(sys.argv[2], "train.pkl")
test_file = os.path.join(sys.argv[2], "test.pkl")

def evaluate(model, matrix, dataset_name):
    """Dump all evaluation metrics and plots for given datasets."""
    eval_path = os.path.join("evaluation", dataset_name)

    labels = matrix[:, 1].toarray().astype(int)
    x = matrix[:, 2:]

    predictions_by_class = model.predict_proba(x)
    predictions = predictions_by_class[:, 1]

    # Use dvclive to log a few simple plots ...
    live = Live(eval_path)
    live.log_plot("roc", labels, predictions)
    live.log("avg_prec", metrics.average_precision_score(labels, predictions))
    live.log("roc_auc", metrics.roc_auc_score(labels, predictions))

    # ... but actually it can be done with dumping data points into a file:
    # ROC has a drop_intermediate arg that reduces the number of points.
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve.
    # PRC lacks this arg, so we manually reduce to 1000 points as a rough estimate.
    precision, recall, prc_thresholds = metrics.precision_recall_curve(labels, predictions)
    nth_point = math.ceil(len(prc_thresholds) / 1000)
    prc_points = list(zip(precision, recall, prc_thresholds))[::nth_point]
    prc_file = os.path.join(eval_path, "plots", "precision_recall.json")
    with open(prc_file, "w") as fd:
        json.dump(
            {
                "prc": [
                    {"precision": p, "recall": r, "threshold": t}
                    for p, r, t in prc_points
                ]
            },
            fd,
            indent=4,
        )


    # ... confusion matrix plot
    live.log_plot("confusion_matrix", labels.squeeze(), predictions_by_class.argmax(-1))


# Load model and data.
with open(model_file, "rb") as fd:
    model = pickle.load(fd)

with open(train_file, "rb") as fd:
    train, feature_names = pickle.load(fd)

with open(test_file, "rb") as fd:
    test, _ = pickle.load(fd)

# Evaluate train and test datasets.
evaluate(model, train, "train")
evaluate(model, test, "test") '''

