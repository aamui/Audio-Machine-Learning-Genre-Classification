import librosa # https://librosa.org/doc/main/feature.html
import librosa.display
from matplotlib import pyplot as plt
import IPython.display as ipd # to play the audio
import ffmpeg
import scipy as sp
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA


song1, sr = librosa.load("data/1.mp3", sr = None, duration=29)
song5, sr = librosa.load("data/5.mp3", sr = None,duration=29)
song77, sr = librosa.load("data/77.mp3", sr = None,duration=29)

# audio of the each clip
ipd.Audio("data/1.mp3")
# ipd.Audio("data/5.mp3")
# ipd.Audio("data/77.mp3")

# plotting wave plot
plt.figure(figsize=(14, 17))
plt.subplot(3,1,1)
plt.title("song1")
librosa.display.waveshow(song1, alpha = 0.5)

plt.subplot(3,1,2)
plt.title("song5")
librosa.display.waveshow(song5, alpha = 0.5)

plt.subplot(3,1,3)
plt.title("song77")
librosa.display.waveshow(song77, alpha = 0.5)

rms_song1.shape

