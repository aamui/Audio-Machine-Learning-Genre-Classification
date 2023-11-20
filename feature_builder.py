import librosa as lib
import sklearn as sk
import numpy as np
from typing import Dict
from scipy import stats


class FeatureBuilder():
    """
    construct a feature tensor of dimension num frames x num songs x num features
    """

    def __init__(self, song_dict: Dict, grads: bool = True, reduce: str = 'None') -> None:
        """

        :param song_dict: dictionary of songs index by nonnegative integers
        :param grad: include estimated derivatives
        :param grad2: include estimated convexity
        :param reduce: apply dimension reduction on features of dim > 1
                       {'PCA', 'MDS', 'LLE', 'Spectral', default = 'None'}
        """
        self.songs = song_dict
        self.dim1_features = self.calc_dim1_features()
        self.multidim_features = self.calc_multidim_features()
        if grads:
            self.dim1_grads = self.calc_dim1_deltas()
            self.multidim_grads = self.calc_multidim_deltas()
        if grads:
            self.feature_tensor = self.dim1_features
            self.feature_tensor = np.append(self.feature_tensor, self.multidim_features, axis=2)
            self.feature_tensor = np.append(self.feature_tensor, self.dim1_grads, axis=2)
            self.feature_tensor = np.append(self.feature_tensor, self.multidim_grads, axis=2)
        else:
            self.feature_tensor = self.dim1_features
            self.feature_tensor = np.append(self.feature_tensor, self.multidim_features, axis=2)



    def calc_dim1_features(self) -> np.ndarray:
        """
        compute features of dim 1 with librosa
        :return: tensor with real entries of dimension num frames * num songs * num features of dimension 1
        """
        dim1_features = np.ndarray(shape=(1025, len(self.songs), 7))
        for i in range(len(self.songs)):
            dim1_features[:, i, 0] = lib.feature.spectral_centroid(y=self.songs[i])[0, :]
            dim1_features[:, i, 1] = lib.feature.spectral_flatness(y=self.songs[i])[0, :]
            dim1_features[:, i, 2] = lib.feature.spectral_rolloff(y=self.songs[i])[0, :]
            dim1_features[:, i, 3] = lib.feature.spectral_bandwidth(y=self.songs[i])[0, :]
            dim1_features[:, i, 4] = lib.feature.zero_crossing_rate(y=self.songs[i])[0, :]
            dim1_features[:, i, 5] = lib.feature.rms(y=self.songs[i])[0, :]
            dim1_features[:, i, 6] = lib.onset.onset_strength(y=self.songs[i])
        return dim1_features


    def calc_dim1_deltas(self) -> np.ndarray:
        """
        approximate first and second derivatives of dim1 features
        :return: tensor with real entries of dimension num frames x num songs x 2 * num features of dimension 1
        """
        dim1_delta = np.ndarray(shape=(1025, len(self.songs), 2 * self.dim1_features.shape[2]))
        for i in range(self.dim1_features.shape[2]):
            for j in range(self.dim1_features.shape[1]):
                dim1_delta[:, j, i] = lib.feature.delta(dim1_features[:, j, i])
        for i in range(self.dim1_features.shape[2], 2*self.dim1_features.shape[2]):
            for j in range(self.dim1_features.shape[1]):
                dim1_delta[:, j, i+self.dim1_features.shape[2]] = lib.feature.delta(dim1_delta[:, j, i])
        return dim1_delta


    def calc_multidim_features(self) -> np.ndarray:
        """
        calculate multidimensional features {MFCCs, Chromas (constant Q and normalized), tonnetz, contrast,
                                             and beats (fits better here I think? might be wrong)}
        :return: tensor with real entries of dimension num frames x num songs x sum(feature * dim of feature)
        """
        multidim_features = np.ndarray(shape=(1025, len(self.songs), 6+7+12+24))
        for i in range(len(self.songs)):
            harm, perc = lib.effects.hpss(self.songs[i])
            #  tempo, beats = lib.beat.beat_track(y=perc)
            #  multidim_features[:, i, 0] = lib.feature.delta(beats)
            multidim_features[:, i, 0:12] = np.transpose(lib.feature.chroma_cens(y=harm))
            multidim_features[:, i, 12:24] = np.transpose(lib.feature.chroma_cqt(y=harm))
            multidim_features[:, i, 24:36] = np.transpose(lib.feature.mfcc(y=harm, n_mfcc=12))
            multidim_features[:, i, 36:42] = np.transpose(lib.feature.tonnetz(y=self.songs[i]))
            multidim_features[:, i, 42:] = np.transpose(lib.feature.spectral_contrast(y=harm))
        return multidim_features


    def calc_multidim_deltas(self) -> np.ndarray:
        """
        approximate first and second derivatives of dim1 features
        :return: tensor with real entries of dimension num frames x num songs x 2 * num components of multidim features
        """
        multidim_delta = np.ndarray(shape=(1025, len(self.songs), 2 * self.multidim_features.shape[2]))
        for i in range(self.multidim_features.shape[2]):
            for j in range(self.dim1_features.shape[1]):
                dim1_delta[:, j, i] = lib.feature.delta(dim1_features[:, j, i])
        for i in range(self.dim1_features.shape[2], 2*self.dim1_features.shape[2]):
            for j in range(self.dim1_features.shape[1]):
                dim1_delta[:, j, i + self.dim1_features.shape[2]] = lib.feature.delta(dim1_delta[:, j, i])
        return dim1_delta


    def reduce_multidim_features(self) -> np.ndarray:
        ...


    @staticmethod
    def summary_stats(feature_tensor: np.ndarray) -> np.ndarray:
        """
        calculate {mean, variance, skewness, kurtosis, min, max} for all features, songs
        :param feature_tensor: num frames x num songs x num features
        :return: features: num summary stats x num songs x num features (6 x 90 x ~50) -> (90 x ~300)
        """
        features = np.ndarray(shape=(6, feature_tensor.shape[1], feature_tensor.shape[2]))
        for i in range(feature_tensor.shape[1]):
            for j in range(feature_tensor.shape[2]):
                features[0, i, j] = np.mean(feature_tensor[:, i, j])
                features[1, i, j] = np.std(feature_tensor[:, i, j])
                features[2, i, j] = stats.skew(feature_tensor[:, i, j])
                features[3, i, j] = stats.kurtosis(feature_tensor[:, i, j])
                features[4, i, j] = min(feature_tensor[:, i, j])
                features[5, i, j] = max(feature_tensor[:, i, j])
        return features