# coding: latin-1
import librosa as rosa
from librosa import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

    return rolling_means.join(rolling_variances)

################## MAIN ################################################################################################

filename = "../bande_son.wav"

# load the audio data
y, sr = rosa.load(filename, sr=11025, mono=True, offset=90, duration=30)

# normalize the audio sample
y = normalize_data(y)
print(y.size) #330750 samples

# Compute MFCC
# Set the hop length; at 22050 Hz, 512 samples ~= 23ms
# Set the hop length; at 11025 Hz, 256 samples ~= 23ms
hop_length = 256  # overlapping (recouvrement)
n_fft = 1024  # (FFT window size) taille de la fenêtre
mfccs = rosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft, n_mels=24, n_mfcc=13)

print(mfccs.shape)  # (13, 130)
# la taille c'est la taille de y / hop_length
# chaque case correspond à un mfcc de 23ms

# Compute RMS energy IN dB
rmse = rosa.feature.rmse(y=y, frame_length=n_fft, hop_length=hop_length)
db = rosa.power_to_db(S=y)

# rolling window integration of short-term features
# compute with rollin window of 100 * 23 ms = 2,3s
long_term_features = integrate(mfccs, window_size=100, rmse=db, logpower_threshold=-20.0)


print type(long_term_features) # pandas.core.frame.DataFrame
print long_term_features.shape  # (1292, 52)
print type(long_term_features.values) # ndarray
print long_term_features.values.T.shape
