'''
Created on 3-lug-2020

@author: Francesco
'''
from utils import serialize_to_file, deserialize_from_file
import numpy as np
import os
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

def map_adaptation(gmm, model_dst, model_name, data, max_iterations, likelihood_threshold = 1e-20, relevance_factor = 16):
    N = data.shape[0]
    D = data.shape[1]
    K = gmm.n_components
    
    mu_k = gmm.means_
    cov_k = gmm.covariances_
    pi_k = gmm.weights_
    
    mu_new = np.zeros((K,D))
    n_k = np.zeros((K,1))

    old_likelihood = 9999999
    new_likelihood = 0
    iterations = 0
    while(abs(old_likelihood - new_likelihood) > likelihood_threshold and iterations < max_iterations):
        iterations += 1
        old_likelihood = new_likelihood
        z_n_k = gmm.predict_proba(data)
        n_k = np.sum(z_n_k,axis = 0)

        for i in range(K):
            temp = np.zeros((1,D))
            for n in range(N):
                temp += z_n_k[n][i]*data[n,:]
            mu_new[i] = (1/(n_k[i]+likelihood_threshold))*temp

        adaptation_coefficient = n_k/(n_k + relevance_factor)
        for k in range(K):
            mu_k[k] = (adaptation_coefficient[k] * mu_new[k]) + ((1 - adaptation_coefficient[k]) * mu_k[k])
        gmm.means_ = mu_k

        log_likelihood = gmm.score(data)
        new_likelihood = log_likelihood
        print(log_likelihood)
    serialize_to_file(gmm, model_dst, model_name)
    
def train_models(features_dir: str, dst_dir: str, ubm):
    files = [f for f in os.listdir(os.path.join(features_dir))]
    speakers = set([])
    for file in files:
        speakers.add(file.split("_")[1])
    print(speakers)
    for sp in sorted(speakers):
        sp_list = [f for f in files if sp in f]
        features = np.asarray(())
        for file in tqdm(sp_list, desc="Deserializing the features files for speaker %s..." % sp):
            vector = deserialize_from_file(os.path.join(features_dir), file)
            if features.size == 0:
                features = vector
            else:
                features = np.vstack((features, vector))
        map_adaptation(ubm, os.path.join(dst_dir), sp+".gmm", features, 15)
        print("Model %s created" % sp)
    

if __name__ == "__main__":
    ubm_file = deserialize_from_file(os.path.join("models", "UBM"), "ubm.gmm")
    train_models(os.path.join("features", "genuine"), os.path.join("models", "ENROLL_genuine"), ubm_file)
    train_models(os.path.join("features", "impostors"), os.path.join("models", "ENROLL_impostors"), ubm_file)
    
    
    
    
    
