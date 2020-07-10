'''
Created on 30 giu 2020

@author: franc
'''
import os
import numpy as np
from scipy.io.wavfile import read
from speakerfeatures import extract_features
import warnings
from sklearn.metrics._classification import precision_score, recall_score
warnings.filterwarnings("ignore")
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from utils import deserialize_from_file
from sklearn.metrics import confusion_matrix

def load_models(models_dir: str):
    """
    Load the gmm models
    Args:
        models_dir (str): Directory where the models are stored
    
    Returns:
        The function returns a tuple containing two lists:
        models: list of the actual models
        speakers: list of the names of the speakers
    """
    gmm_files = os.listdir(os.path.join(models_dir))
    models = [deserialize_from_file(models_dir, fname) for fname in gmm_files]
    speakers = [fname.split(".gmm")[0] for fname in gmm_files]
    return (models, speakers)

def load_test_audios(test_audio_dir: str):
    """
    Extracts the features from the test files
    Args:
        test_audio_dir: Path to the directory in which the test audios are stored
    Returns
        vectors (list): List of features vectors, one for each audio
    """
    audio_tests = os.listdir(os.path.join(test_audio_dir))
    vectors = []
    for file in audio_tests:
        sr,audio = read(os.path.join(test_audio_dir, file))
        vector = extract_features(audio,sr)
        vectors.append((file.split('.')[0], vector))
    return vectors
    
def test(models, speakers, ubm_dir, ubm_name, features_vectors, df_path, df_name):
    """
    This function does the following things:
      1) Deserialize the ubm and add that model to the speakers list
      2) For each feature vector, confront it with all the models. The model with the highest
         likelihood ratio will be the predicted model
      3) Create a confusion matrix cm, where cm(i, j) is the number of times that the i-th model
         has been recognized as j-th model
      4) Creates a pandas DataFrame containing one column for the test audios and one for the model
         that has been predicted for those test audios
    
    Args:
        models (list): List of gmm models
        speakers (list): List of speakers
        ubm_dir (str): Path to the directory where the ubm model is stored
        ubm_name (str): Name of the file containing the ubm model
        features_vectors: List of features vectors extracted from the test audios
        df_path (str): Path to the directory where the dataframe will be saved as xlsx. If it doesn't
           exist it will be created
        df_name: Name of the file in which save the dataframe
    Returns:
        
    """
    ubm = deserialize_from_file(os.path.join(ubm_dir), ubm_name)
    models.append(ubm)
    speakers.append("ubm")
    y_true = []     #For the confuzion matrix
    y_pred = []     #For the confuzion matrix
    rows_list = []  #For the dataframe
    
    for t in features_vectors:
        print(t[0])
        vector = t[1]
        log_likelihood = np.zeros(len(models)) 
        
        for i in range(len(models)):
            gmm    = models[i]         #checking with each model one by one
            scores = np.array(gmm.score(vector))
            log_likelihood[i] = scores.sum()
        
        winner = np.argmax(log_likelihood)
        dict = {}
        dict["Audio"] = t[0]
        dict["Detected_as"] = speakers[winner]
        rows_list.append(dict)
        y_true.append(t[0].split('_')[1])
        y_pred.append(speakers[winner])
    df = pd.DataFrame(rows_list)
    
    if not os.path.exists(df_path):
        os.makedirs(df_path)
    df.to_excel(os.path.join(df_path, df_name), index=False)
    
    return (y_true, y_pred)

def compute_precision_recall(y_true, y_pred):
    return (precision_score(y_true, y_pred, average=None) * 100, recall_score(y_true, y_pred, average=None) * 100)

def plot_matrix(y_true, y_pred, labels, img_path, img_name):
    cm = confusion_matrix(y_true, y_pred)
    
    #RESHAPE IF NEEDED
    while len(labels) > cm.shape[0]:
        cm = np.vstack([cm, np.zeros(cm.shape[1])])
    while len(labels) > cm.shape[1]:
        y = np.array(np.zeros(cm.shape[0]))
        y = y.reshape(cm.shape[0], 1)
        cm = np.append(cm, y, axis=1)
        
    df_cm = pd.DataFrame(cm, index=[i for i in labels], columns=[i[:3] for i in labels])
    fig = plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    fig.suptitle(img_name.split('.')[0])
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    #plt.show()
    
    #Build the path for saving the image if the folders given don't exist
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    fig.savefig(os.path.join(img_path, img_name))

if __name__ == "__main__":
    ##### BASELINE CASE #####
    models, speakers = load_models(os.path.join("models", "ENROLL_genuine"))
    
    #TEST AUDIOS 1S LONG
    #vectors = load_test_audios(os.path.join("dataset", "test_1s", "genuine"))
    #y_true, y_pred = test(models, speakers, os.path.join("models", "UBM"), "ubm.gmm", vectors, "results/dataframes", "baseline_1s.xlsx")
    #plot_matrix(y_true, y_pred, speakers, "results/plots", "Baseline_1s.png")
    #print(compute_precision_recall(y_true, y_pred))
    
    #TEST AUDIOS 5S LONG
    #vectors = load_test_audios(os.path.join("dataset", "test_5s", "genuine"))
    #y_true, y_pred = test(models, speakers, os.path.join("models", "UBM"), "ubm.gmm", vectors, "results/dataframes", "baseline_5s.xlsx")
    #plot_matrix(y_true, y_pred, speakers, "results/plots", "Baseline_5s.png")
    #print(compute_precision_recall(y_true, y_pred))
    
    #TEST AUDIOS 20S LONG
    vectors = load_test_audios(os.path.join("dataset", "test_20s", "genuine"))
    #y_true, y_pred = test(models, speakers, os.path.join("models", "UBM"), "ubm.gmm", vectors, "results/dataframes", "baseline_20s.xlsx")
    #plot_matrix(y_true, y_pred, speakers, "results/plots", "Baseline_20s.png")
    #print(compute_precision_recall(y_true, y_pred))
    
    
    #### SECOND CASE ######
    models1, speakers1 = load_models(os.path.join("models", "ENROLL_impostors"))
    models = models + models1
    speakers = speakers + speakers1
    
    #AUDIO TESTS 1S LONG
    #vectors1 = load_test_audios(os.path.join("dataset", "test_1s", "impostors"))
    #vectors = vectors + vectors1
    #y_true, y_pred = test(models, speakers, os.path.join("models", "UBM"), "ubm.gmm", vectors, "results/dataframes", "Second_1s.xlsx")
    #plot_matrix(y_true, y_pred, speakers, "results/plots", "Second_1s.png")
    #print(compute_precision_recall(y_true, y_pred))
    
    #AUDIO TESTS 5S LONG
    #vectors1 = load_test_audios(os.path.join("dataset", "test_5s", "impostors"))
    #vectors = vectors + vectors1
    #y_true, y_pred = test(models, speakers, os.path.join("models", "UBM"), "ubm.gmm", vectors, "results/dataframes", "Second_5s.xlsx")
    #plot_matrix(y_true, y_pred, speakers, "results/plots", "Second_5s.png")
    #print(compute_precision_recall(y_true, y_pred))
    
    #AUDIO TESTS 20S LONG
    """vectors1 = load_test_audios(os.path.join("dataset", "test_20s", "impostors"))
    vectors = vectors + vectors1"""
    #y_true, y_pred = test(models, speakers, os.path.join("models", "UBM"), "ubm.gmm", vectors, "results/dataframes", "Second_20s.xlsx")
    #plot_matrix(y_true, y_pred, speakers, "results/plots", "Second_20s.png")
    #print(compute_precision_recall(y_true, y_pred))
    
    
    #### IMITATION CASE ####
    #AUDIO TESTS 1S LONG
    #vectors_imitations = load_test_audios(os.path.join("dataset", "test_1s", "attacks"))
    #vectors = vectors + vectors_imitations
    #y_true, y_pred = test(models, speakers, os.path.join("models", "UBM"), "ubm.gmm", vectors, "results/dataframes", "Imitation_1s.xlsx")
    #plot_matrix(y_true, y_pred, speakers, "results/plots", "Imitations_1s.png")
    #print(compute_precision_recall(y_true, y_pred))
    
    #AUDIO TESTS 5S LONG
    #vectors_imitations = load_test_audios(os.path.join("dataset", "test_5s", "attacks"))
    #vectors = vectors + vectors_imitations
    #y_true, y_pred = test(models, speakers, os.path.join("models", "UBM"), "ubm.gmm", vectors, "results/dataframes", "Imitation_5s.xlsx")
    #plot_matrix(y_true, y_pred, speakers, "results/plots", "Imitations_5s.png")
    #print(compute_precision_recall(y_true, y_pred))
    
    #AUDIO TESTS 20S LONG
    vectors_imitations = load_test_audios(os.path.join("dataset", "test_20s", "attacks"))
    vectors = vectors + vectors_imitations
    y_true, y_pred = test(models, speakers, os.path.join("models", "UBM"), "ubm.gmm", vectors, "results/dataframes", "Imitations_20s.xlsx")
    plot_matrix(y_true, y_pred, speakers, "results/plots", "Imitations_20s.png")
    print(compute_precision_recall(y_true, y_pred))