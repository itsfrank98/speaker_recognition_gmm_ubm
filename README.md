# speaker_recognition_gmm_ubm
A system that adopts GMM_UBM to recognize a speaker. Tested on imitations

# Description
This Python system implements a GMM_UBM approach to model its speakers. Its aim is to distinguish between a genuine speaker and an attacker who imitates his voice to fraud the system. It's tested on a dataset of italian voices that I created. The process consistes of the following steps:
1. **Preprocessing**: The audios are downsampled to 8kHz, converted to wav in case they aren't in this format and chunked in pieces 20 seconds long
2. **Features extraction** for both the model audios and the UBM audios
3. **UBM training**. The datas for the UBM training were downloaded from here --> https://voice.mozilla.org/it/datasets. I used the italian dataset, that contains more than 80k files. I reduced them to be less than 1000 or the training phase would have took much more time (it took 2 hours with 803 files). The training of the UBM model was done using thr Google Colab notebook.
4. **Model training** using MAP adaptation from the UBM.
5. **Testing**: The testing consisted in three phases:
  1. In the first case, we assume that the system only knows the genuine speakers, and is tested only on the audios belonging to them. 
  2. In the second case, we assume that the system knows both the genuine speakers and their imitator's normal voice.
  3. In the third case the system knows the genuine speakers and their imitator's normal voices. The testing is done using audios belonging to them, and are used the 
  audios of the imitations too. This approach was adopted in the paper contained in the "papers" folder.
  Other than that, the test audios were chunked in pieces 5-seconds long and then again in pieces 1-second long. The three phases above were repeated using first the 
  test audios 20 seconds long, then the audios 5-seconds long and lastly the audios 1-second long. As you can imagine, the longer the test audios, the better accuracy.
  <br>

The results are in the results/plots folder, and were pretty good but we have to consider the small size of the dataset.

## Dataset structure
The dataset has this structure: <br>
```
|-- dataset
|    |-- enroll
|    |      |-- genuine
|    |      |-- impostors
|    |-- test_1s
|    |      |-- attacks
|    |      |-- genuine
|    |      |-- impostors
|    |-- test_5s<br>
|    |      |-- attacks
|    |      |-- genuine
|    |      |-- impostors
|    |-- test_20s<br>
|    |      |-- attacks
|    |      |-- genuine
|    |      |-- impostors
|    |-- ubm
```

**The audio names structure is the following:**
* For files in the enroll folders:  ENROLL_SpeakerName_ChunkNumber.wav (for example: ENROLL_Amadeus_1.wav for speaker Amadeus)
* For files in test folders (but **NOT** the "attacks" folder): TEST_SpeakerName_ChunkNumber.wav (for example: TEST_Amadeus_1.wav for speaker Amadeus)
* For files in the attacks folder: TEST_ImitatorName_GenuineSpeakerName_ChunkNumber.wav (for example, for the first audio in which the speaker "Fiorello" imitates speaker "Bongiorno" the audios name are: TEST_Fiorello_Bongiorno_1.wav)
 
## Requirements:
The following Python libraries were used:
* librosa~=0.7.2
* matplotlib~=3.2.2
* numpy~=1.19.0
* pandas~=1.0.5
* openpyxl~=3.0.4
* pydub~=0.24.1
* python-speech-features~=0.6
* scikit-learn~=0.23.1
* scipy~=1.5.0
* seaborn~=0.10.1
* tqdm~=4.47.0
* sklearn
* pickle-mixin~=1.0.2
<br>
Python may ask you to install other dependencies if you don't have them.


## Credits:
* The features extraction, UBM training and part of the final testing were done by adapting the code provided by @abhijeet3922. The code is [here](https://github.com/abhijeet3922/Speaker-identification-using-GMMs), and it's discussed in his blog [here](https://appliedmachinelearning.blog/2017/11/14/spoken-speaker-identification-based-on-gaussian-mixture-models-python-implementation/).
* The MAP adaptation for the speakers models was done by using part of the code provided by @scelesticsiva in [this file](https://github.com/scelesticsiva/speaker_recognition_GMM_UBM/blob/master/src/speaker_recognition/MAP_adapt.py).

