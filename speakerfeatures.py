'''
Created on 30 giu 2020

@author: franc
'''
import numpy as np
from sklearn import preprocessing
import python_speech_features as mfcc
from utils import serialize_to_file
import os

def calculate_delta(array):
    """Calculate and returns the delta of given feature vector matrix"""

    rows,cols = array.shape
    deltas = np.zeros((rows,20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i-j < 0:
                first = 0
            else:
                first = i-j
            if i+j > rows -1:
                second = rows -1
            else:
                second = i+j
            index.append((second,first))
            j+=1
        deltas[i] = (array[index[0][0]] - array[index[0][1]] + (2 * (array[index[1][0]] - array[index[1][1]]))) / 10
    return deltas

def extract_features(audio,rate):
    """extract 20 dim mfcc features from an audio, performs CMS and combines 
    delta to make it 40 dim feature vector"""    
    
    mfcc_feat = mfcc.mfcc(audio, rate, 0.025, 0.01, 20, appendEnergy=True)
    
    mfcc_feat = preprocessing.scale(mfcc_feat)
    delta = calculate_delta(mfcc_feat)
    combined = np.hstack((mfcc_feat,delta))
    return combined

def feature_extraction(src, dst):
    from scipy.io.wavfile import read
    from tqdm import tqdm
    
    if not os.path.isdir(os.path.join(dst)):
        os.mkdir(os.path.join(dst))
    
    files = os.listdir(os.path.join(src))
    for file in tqdm(files, desc="Extracting features"):
        sr, audio = read(os.path.join(src, file))
        vector = extract_features(audio,sr)
        serialize_to_file(vector, dst, file.split('.')[0])
    
#    
if __name__ == "__main__":
    ##### FEATURES EXTRACTION FOR GENUINE ENROLLMENT AUDIOS ####
    feature_extraction("dataset/enroll/genuine", "features")
    
    ##### FEATURES EXTRACTION FOR IMPOSTORS ENROLLMENT AUDIOS ####
    feature_extraction("dataset/enroll/impostors", os.path.join("features", "impostors"))

    ##### EXTRACTION FEATURES FOR UBM AUDIOS ####
    feature_extraction("dataset/ubm")
    
    
    
    