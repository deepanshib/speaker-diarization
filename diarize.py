# coding: latin-1
import librosa as rosa
from librosa import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def normalize_data(data):
    normalized = data - data.mean()
    normalized = normalized / normalized.max()
    return normalized


# takes in input an ndarray, returns rolling [original.mean, derivative.mean, acceleration.mean, original.variance]
#  with a centered rolling window of size window size
def integrate(mfccs, window_size, rmse, logpower_threshold = None):
    # compute first & second derivative
    derivative = rosa.feature.delta(mfccs, order=1)  # derivative
    acceleration = rosa.feature.delta(mfccs, order=2)  # second derivative

    # use panda library to join vectors & compute rolling means & variances
    df_mfcc = pd.DataFrame(data=mfccs.transpose()).add_prefix('MFCC_')
    df_derivative = pd.DataFrame(data=derivative.transpose()).add_prefix('DERIV_')
    df_accel = pd.DataFrame(data=acceleration.transpose()).add_prefix('ACCEL_')

    # set values corresponding to silence to n/a ( i.e. <  logpower_threshold (in dB )
    # so they won't participate in the rolling mean/var computation.
    df_rmse = pd.Series(data=rmse.transpose())

    if logpower_threshold is not None:
        df_mfcc.where(df_rmse > logpower_threshold, inplace=True)
        df_derivative.where(df_rmse > logpower_threshold, inplace=True)
        df_accel.where(df_rmse > logpower_threshold, inplace=True)

    rolling_means = df_mfcc.join(df_derivative).join(df_accel).rolling(window=window_size,
                                                                       center=True, min_periods=1).mean().add_prefix('MEAN_')
    rolling_variances = df_mfcc.rolling(window=window_size, center=True, min_periods=1).var().add_prefix('VAR_')

    longterm_features = rolling_means.join(rolling_variances)

    longterm_features = (longterm_features - longterm_features.mean()) / longterm_features.std()
    return longterm_features


def compute_longterm_features(y, sr, offset, hop_length, n_fft, n_mels=24, n_mfcc=13, logpower_threshold=None):

    # normalize the audio sample
    y = normalize_data(y)

    # compute mfcss
    mfccs = rosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft, n_mels=n_mels, n_mfcc=n_mfcc)

    #print(mfccs.shape)  # (13, 130)
    # la taille c'est la taille de y / hop_length
    # chaque case correspond à un mfcc de 23ms

    #  Compute RMS energy IN dB
    rmse = rosa.feature.rmse(y=y, frame_length=n_fft, hop_length=hop_length)
    db = rosa.power_to_db(S=y)

    # rolling window integration of short-term features
    # compute with rollin window of 100 * 23 ms = 2,3s
    longterm_features = integrate(mfccs, window_size=50, rmse=db, logpower_threshold=logpower_threshold)

    #  add timeline index
    # sample  duration = 1.0 / sr * hop_length
    longterm_features['time'] = (1.0 / sr * hop_length) * np.arange(mfccs.shape[1]) + offset
    longterm_features.dropna(inplace=True)

    return longterm_features

# fonction qui applique la méthode des k means aux fenêtres Long terme avec k clusters
def classifk(k,fenetresLT):
    akai=0
    d = np.shape(fenetresLT)
    kmeans=KMeans(n_clusters=k)
    # ici on fite sur les fenetres
    kmeans.fit(fenetresLT)
    # ici on détermine à quel cluster appartient chaque fenêtre
    locuteur=kmeans.predict(fenetresLT)
    # kmeans.interia mesure l'inertie au sein des clusters, donc la partie de l'inertie totale que les clusters n'expliquent pas
    akai=kmeans.inertia_
    # donne les coordonnées des centres des clusters
    ousontils=kmeans.cluster_centers_
    return locuteur,akai,ousontils



################## MAIN ################################################################################################

filename = "../bande_son.wav"

# load the audio data
offset=90
duration=30

offset=2*60+34
duration=2250

y, sr = rosa.load(filename, sr=22050, mono=True, offset=offset, duration=duration)


#print(y.size) #330750 samples

# params for MFCC
# Set the hop length; at 22050 Hz, take 512 samples ~= 23ms
# Set the hop length; at 11025 Hz, take 256 samples ~= 23ms
hop_length = 512  # overlapping (recouvrement)
n_fft = 1024  # (FFT window size) taille de la fenêtre

# valeur du paramètre de seuillage
logpower_threshold=None

longterm_features = compute_longterm_features(y, sr, offset=offset, hop_length=hop_length, n_fft=n_fft, logpower_threshold=logpower_threshold)

# on essaie la méthode des k means en faisant varier le nombre de clusters de 1 à 10 (en fait on sait qu'il y a 5 locuteurs dans l'émission)
k=0
ac=np.zeros(16)
percent=np.zeros(16)
for k in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]:
#   lo,ac[k],ou = classifk(k,longterm_features.iloc[:,:-1])
    lo,ac[k],ou = classifk(k,longterm_features.iloc[:,:13])
    percent[k]=ac[k]/ac[1]
# ac donne la dispersion intracluster en fonction du nombre de clusters
# ac[1] est la dispersion totale
print(ac)
# percent donne la part de la dispersion totale expliquée par la dispersion intracluster, en fonction du nombre de cluster
# (autrement dit ce que les clusters n'expliquent pas. Ca chute avec leur nombre)
print(percent)

# on calcule le gain d'explication de la dispersion totale apporté par l'ajout d'un cluster supplémentaire
# et on infère que le nombre de locuteurs kk est celui pour lequel le gain marginal d'un cluster supplémentaire est maximal
kk=1
marginal=0.0
for k in [2,3,4,5,6,7,8,9,10,11,12,13,14,15]:
    if marginal < percent[k-1]-percent[k]:
        kk=k
        marginal=percent[k-1]-percent[k]
    print('gain marginal de l\'ajout d\'un ',k,'-ième cluster : ','{:.1%}'.format(percent[k-1]-percent[k]))
print('Le système reconnait ', kk,' (groupe de) locuteurs')