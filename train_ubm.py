'''
Created on 30 giu 2020

@author: franc
'''
from tqdm import tqdm
import numpy as np
import os
from sklearn.mixture import GaussianMixture
from utils import serialize_to_file, deserialize_from_file
import warnings
warnings.filterwarnings("ignore")


def train_models(features_dir: str, group: str, dst: str):
    """
    Train the models for the audios in the enrollment and test set
    """
    group = group.upper()
    if not os.path.isdir(dst):
        os.mkdir(dst)
    os.chdir(dst)
    if not os.path.isdir(group):
        os.mkdir(group)
    os.chdir('..')
    
    assert group in ("ENROLL", "UBM"), "Group must either be enroll or UBM!"
    files = [f for f in os.listdir(os.path.join(features_dir)) if f.startswith(group)]
    if group == "UBM":
        speakers = files
        train_model(features_dir, speakers, "ubm", os.path.join(dst, group))
    
        
            
            
def train_model(features_dir: str, files: list, model_name: str, dst: str):
    """
    Train a single model
    Args:
        features_dir: Path to the directory in which the features files are stored
        files: list of filenames to use for training the model
        model_name: Name of the model to create
        dst: Path for the directory in which the model will be serialized
    """
    features = np.asarray(())
    
    #Take the feature vectors needed for training the model
    for file in tqdm(files, desc="Deserializing the features files..."):
        vector = deserialize_from_file(features_dir, file)
        if features.size == 0:
            features = vector
        else:
            features = np.vstack((features, vector))
    
    #Train GMM using the features vectors retrieved
    gmm = GaussianMixture(n_components = 512, max_iter = 200, covariance_type='diag',n_init = 3)
    gmm.fit(features)
    picklefile = model_name + ".gmm"
    serialize_to_file(gmm, os.path.join(dst), picklefile)
    print('+ modeling completed for speaker:', model_name, " with data point = ", features.shape)

if __name__ == "__main__":
    train_models("features", "ubm", "models")
    
    