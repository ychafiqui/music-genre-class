import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import librosa
import os
import base64
import audio2numpy as a2n

os.environ["PATH"] += os.pathsep + 'C:/Program Files/ffmpeg/bin' # to be removed in production

columns = ['chroma_stft_mean', 'chroma_stft_var', 'rms_mean', 'rms_var', 'spectral_centroid_mean', 'spectral_centroid_var',
           'spectral_bandwidth_mean', 'spectral_bandwidth_var', 'rolloff_mean', 'rolloff_var', 'zero_crossing_rate_mean', 
           'zero_crossing_rate_var', 'harmony_mean', 'harmony_var', 'perceptr_mean', 'perceptr_var', 'tempo', 'mfcc1_mean', 
           'mfcc1_var', 'mfcc2_mean', 'mfcc2_var', 'mfcc3_mean', 'mfcc3_var', 'mfcc4_mean', 'mfcc4_var', 'mfcc5_mean', 
           'mfcc5_var', 'mfcc6_mean', 'mfcc6_var', 'mfcc7_mean', 'mfcc7_var', 'mfcc8_mean', 'mfcc8_var', 'mfcc9_mean', 
           'mfcc9_var', 'mfcc10_mean', 'mfcc10_var', 'mfcc11_mean', 'mfcc11_var', 'mfcc12_mean', 'mfcc12_var', 
           'mfcc13_mean', 'mfcc13_var', 'mfcc14_mean', 'mfcc14_var', 'mfcc15_mean', 'mfcc15_var', 'mfcc16_mean', 
           'mfcc16_var', 'mfcc17_mean', 'mfcc17_var', 'mfcc18_mean', 'mfcc18_var', 'mfcc19_mean', 'mfcc19_var', 
           'mfcc20_mean', 'mfcc20_var', 'label']

dataset = pd.read_csv("music_genre_dataset_30s.csv")
ss = StandardScaler()
le = LabelEncoder()
le.fit_transform(dataset["label"])

def get_dataset_normilized():
    dataset = ss.fit_transform(dataset)
    return dataset, ss

def normilize_newdata(new_data):
    test_frame = pd.DataFrame([new_data], columns=columns[:-1])
    train_plus_test = pd.concat([dataset, test_frame])
    train_plus_test = train_plus_test.drop(columns=["label"])
    train_plus_test_norm = ss.fit_transform(train_plus_test)
    test_frame_norm = train_plus_test_norm[-1]
    test_frame_norm = np.array([test_frame_norm])
    return test_frame_norm

def decode_label(label):
    return le.inverse_transform(label)[0]

def save_file(name, content):
    """Decode and store a file uploaded with Plotly Dash."""
    data = content.encode("utf8").split(b";base64,")[1]
    with open(os.path.join("audio_files", name), "wb") as fp:
        fp.write(base64.decodebytes(data))

def extract_features_audio_file(file, offset=0, duration=60):
    if file.endswith(".mp3"):
        x, sr = a2n.audio_from_file(file, offset=offset, duration=duration)
        audio_data = x.T[0]
    elif file.endswith(".wav"):
        audio_data, sr = librosa.load(file, offset=offset, duration=duration)
    test_data = []
    
    #chroma
    S = np.abs(librosa.stft(audio_data, n_fft=4096))**2
    chroma = librosa.feature.chroma_stft(S=S, sr=sr)
    #chroma_stft_mean
    chroma_mean = round(np.mean(chroma),6)
    test_data.append(chroma_mean)
    #chrome_stft_var
    chroma_var = round(np.var(chroma),6)
    test_data.append(chroma_var)

    #rms
    rms = librosa.feature.rms(y=audio_data)
    #rms_mean
    rms_mean = round(np.mean(rms),6)
    test_data.append(rms_mean)
    #rms_var
    rms_var = round(np.var(rms),6)
    test_data.append(rms_var)

    #spectral_centroid
    cent = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
    #spectral_centroid_mean
    sc_mean = round(np.mean(cent),6)
    test_data.append(sc_mean)
    #spectral_centroid_var
    sc_var = round(np.var(cent),6)
    test_data.append(sc_var)

    #spectral_bandwidth
    spec_bw = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)
    #spectral_bandwidth_mean
    spec_bw_mean = round(np.mean(spec_bw),6)
    test_data.append(spec_bw_mean)
    #spectral_bandwidth_var
    spec_bw_var = round(np.var(spec_bw),6)
    test_data.append(spec_bw_var)

    #rolloff
    rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
    #rolloff_mean
    rolloff_mean = round(np.mean(rolloff),6)
    test_data.append(rolloff_mean)
    #rolloff_var
    rolloff_var = round(np.var(rolloff),6)
    test_data.append(rolloff_var)

    #zero_crossing_rate
    zcr = librosa.feature.zero_crossing_rate(audio_data)
    #zero_crossing_rate_mean
    zcr_mean = round(np.mean(zcr),6)
    test_data.append(zcr_mean)
    #zero_crossing_rate_var
    zcr_var = round(np.var(zcr),6)
    test_data.append(zcr_var)

    # harmony and perceptr
    y_harm, y_perc = librosa.effects.hpss(audio_data)
    #harmony_mean
    harmony_mean = round(np.mean(y_harm),6)
    test_data.append(harmony_mean) 
    #harmony_var
    harmony_var = round(np.var(y_harm),6)
    test_data.append(harmony_var)
    #perceptr_mean
    perceptr_mean = round(np.mean(y_perc),6)
    test_data.append(perceptr_mean) 
    #perceptr_var
    perceptr_var = round(np.var(y_perc),6)
    test_data.append(perceptr_var)
    #tempo
    hop_length = 512
    oenv = librosa.onset.onset_strength(y=audio_data, sr=sr, hop_length=hop_length)
    tempo = librosa.beat.tempo(onset_envelope=oenv, sr=sr, hop_length=hop_length)[0]
    tempo = round(tempo,6)
    test_data.append(tempo)
    
    d = librosa.feature.mfcc(y=np.array(audio_data).flatten(), sr=sr , n_mfcc=20)
    d_var = d.var(axis=1).tolist()
    d_mean = d.mean(axis=1).tolist()
    for i in range(20):
        test_data.append(d_mean[i])
        test_data.append(d_var[i])
    return test_data