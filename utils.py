'''
Created on 1 lug 2020

@author: franc
'''
import pickle
import os
import librosa
from tqdm import tqdm
from pydub import AudioSegment
from scipy.io.wavfile import read, write

def serialize_to_file(content, dst, file_name):
    if not os.path.isdir(os.path.join(dst)):
        os.mkdir(os.path.join(dst))
    f = open(dst+"/"+file_name, "wb")
    pickle.dump(content, f)
    f.close()

def deserialize_from_file(src, file_name):
    f = open(os.path.join(src, file_name),'rb')
    content = pickle.load(f)
    f.close()
    return content

def vad():
    print("Reading in the wave file...")
    #AudioSegment.converter = "C:\ffmpeg"
    seg = AudioSegment.from_wav(os.path.join('ENROLL_Amadeus_3.wav'))
    print("Detecting voice...")
    results = seg.detect_voice()
    voiced = [tup[1] for tup in results if tup[0] == 'v']
    unvoiced = [tup[1] for tup in results if tup[0] == 'u']
    
    print("Reducing voiced segments to a single wav file 'voiced.wav'")
    voiced_segment = voiced[0].reduce(voiced[1:])
    voiced_segment.export("voiced.wav", format="WAV")
    
    print("Reducing unvoiced segments to a single wav file 'unvoiced.wav'")
    unvoiced_segment = unvoiced[0].reduce(unvoiced[1:])
    unvoiced_segment.export("unvoiced.wav", format="WAV")
    

def convert_wav():
    """
    Utility function to convert the audio in the ubm, that are in mp3 format
    """  
    os.chdir("dataset/ubm/clips")
    for file in os.listdir():
        os.system(f"""ffmpeg -i {file} -acodec pcm_u8 -ar 22050 {file[:-4]}.wav""")

def resample(src, new_frequency):
    files = os.listdir(os.path.join(src))
    for file in tqdm(files, desc="Resampling files"):
        y, sr = librosa.load(os.path.join(src, file), sr=new_frequency, mono=True)
        librosa.output.write_wav(os.path.join(src, file), y, sr)
        
def split_audios(chunk_size, src_directory, dst_directory):
    """
    Goes into directory where the audios are stored and splits them in chunks. 
    If to_set == "test" then the split is made only once. If to_set == "enrollment" the split 
    is made iteratively    
    """
    if not os.path.isdir(os.path.join(dst_directory)):
        os.mkdir(os.path.join(dst_directory))
    audio_chunks = []
    for file in tqdm(os.listdir(os.path.join(src_directory)), desc="Chunking audios"):
        file_name = file.split('.')[0]
        audio = AudioSegment.from_file(os.path.join(src_directory, file))
        
        if chunk_size > len(audio):
            raise ValueError("The chunk size is larger than the length of the audio!")
        t1 = 0
        t2 = chunk_size
        i = 1
        while t2 <= len(audio):
            part = audio[t1:t2]
            audio_chunks.append((file_name+'_'+str(i), part))
            t1 = t2
            t2 += chunk_size
            i += 1
    
    #Saves the audios in the dst directory
    for file, audio in audio_chunks:
        #write(os.path.join(dst_directory, file+".wav"), 8000, audio)
        audio.export(os.path.join(dst_directory, file+".wav"), format="wav")
  
if __name__ == "__main__":
    #### RESAMPLE THE AUDIOS TO 8KHz FREQUENCY ####
    resample("dataset/enroll/genuine", 8000)
    resample("dataset/enroll/impostors", 8000)
    resample("dataset/test/genuine", 8000)
    resample("dataset/test/impostors", 8000)
    resample("dataset/ubm", 8000)
    resample("dataset/test/attacks", 8000)
    
    #### CHUNK THE TEST AUDIOS ####
    split_audios(1000, os.path.join("dataset", "test", "attacks"), os.path.join("test_1s", "attacks"))
    split_audios(1000, os.path.join("dataset", "test", "genuine"), os.path.join("test_1s", "genuine"))
    split_audios(1000, os.path.join("dataset", "test", "impostors"), os.path.join("test_1s", "impostors"))
    
    split_audios(5000, os.path.join("dataset", "test", "attacks"), os.path.join("test_5s", "attacks"))
    split_audios(5000, os.path.join("dataset", "test", "genuine"), os.path.join("test_5s", "genuine"))
    split_audios(5000, os.path.join("dataset", "test", "impostors"), os.path.join("test_5s", "impostors"))
    
    
    
