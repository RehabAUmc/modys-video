from pathlib import Path
from typing import List, Tuple

import numpy as np
from keras.utils.data_utils import Sequence

from src.features import get_dtmp_distribution_statistics, get_dtl_distribution_statistics, \
    get_angle_distribution_statistics
from src.helpers import read_video
from src.settings import LYING_VIDEOS_DATA_FOLDER

VALID_BODYPARTS = ['ankle', 'knee', 'hip', 'wrist', 'elbow', 'shoulder', 'forehead', 'chin']
DEFAULT_BODYPARTS_RAW_GENERATOR = ['ankle1', 'knee1', 'hip1', 'ankle2', 'knee2', 'hip2']

class FeatureConfiguration:
    def __init__(self, dmtp_bodyparts: List[str] = None, dtl_bodyparts: List[str] = None,
                 angle_bodypart_triples: List[Tuple[str]] = None):
        """
        Configuration for features to use. Allowed body parts:
            ['ankle', 'knee', 'hip', 'wrist', 'elbow', 'shoulder', 'forehead', 'chin']

        Args:
            dmtp_bodyparts: body parts for which to compute distance to middle point (dmtp)
                features, for example: ['ankle', 'hip', 'knee']
            dtl_bodyparts: body parts for which to compute distance to line (dtl) features,
                for example: ['ankle', 'knee']
            angle_bodypart_triples: a list of triples of body parts, for each triple in the list
                the angle between the first and list bodypart is calculated with the middle body
                part as joint. For example [('hip', 'knee', 'ankle')] will calculate 1 angle
                between the hip and the ankle with the knee as joint.
        """
        if dmtp_bodyparts is None:
            dmtp_bodyparts = []
        self.dmtp_bodyparts = dmtp_bodyparts
        print('Computing distance to middlepoint features on: ',
              dmtp_bodyparts)

        if dtl_bodyparts is None:
            dtl_bodyparts = []
        self.dtl_bodyparts = dtl_bodyparts
        print('Computing distance to line features on: ',
              dtl_bodyparts)

        if angle_bodypart_triples is None:
            angle_bodypart_triples = []
        self.angle_bodypart_triples = angle_bodypart_triples
        print('Computing angle features on: ',
              angle_bodypart_triples)

        assert all(self._is_valid_bodypart(b) for b in dmtp_bodyparts)
        assert all(self._is_valid_bodypart(b) for b in dtl_bodyparts)
        assert all(self._is_valid_bodypart(b) for bodyparts in angle_bodypart_triples
                   for b in bodyparts)

    @staticmethod
    def _is_valid_bodypart(bodypart: str):
        return bodypart in VALID_BODYPARTS


class DataGeneratorBase(Sequence):
    def __init__(self, scores_df, batch_size=1, videos_folder=LYING_VIDEOS_DATA_FOLDER):
        self.batch_size = batch_size
        # Drop 'ID' column, we needed that for grouped cross validation splits, but if we drop it
        # we only have scores in our dataframe.
        if 'ID' in scores_df:
            scores_df = scores_df.drop(columns='ID')
        self.scores_df = scores_df
        self.indexes = self.scores_df.index
        self.videos_folder = videos_folder
        if not Path(videos_folder).exists():
            raise FileNotFoundError('The path to videos folder does not exist')

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.indexes) / self.batch_size))

    def _generate_y(self, indexes):
        return self.scores_df.loc[indexes].values

    def _generate_X(self, indexes):
        raise NotImplementedError()

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        # Generate data
        X = self._generate_X(indexes)
        y = self._generate_y(indexes)
        return X, y


class RawDataGenerator(DataGeneratorBase):
    """
    Data generator that yields raw coordinate data (so no handcrafted features).

    Args:
        scores_df: pd.Dataframe with video id as index and the different score columns (in form
            'D_LLP_R_tA_pscore').
        batch_size: number of samples per batch
        videos_folder: folder where videos are located
        drop_likelihood: toggles dropping the likelihood column
        scaler: Scikitlearn skaler to apply to data
        cutoff: cutoff this number of frames from the start
        interpolation_threshold: If the likelihood falls below this threshold we will interpolate
            coordinates
        bodyparts: bodyparts to include coordinates from (in form ankle1,
        input_sequence_len: number of frames to include (before cutting off frames from the start)
    """
    def __init__(self, scores_df, batch_size=1, videos_folder=LYING_VIDEOS_DATA_FOLDER,
                 drop_likelihood=True, scaler=None, cutoff=0,
                 interpolation_threshold=None,
                 bodyparts=None,
                 input_sequence_len=501):
        super().__init__(scores_df, batch_size, videos_folder)
        self.drop_likelihood = drop_likelihood
        self.input_sequence_len = input_sequence_len
        self.scaler = scaler
        self.cutoff = cutoff
        self.interpolation_threshold = interpolation_threshold
        self.bodyparts = bodyparts if bodyparts is not None else DEFAULT_BODYPARTS_RAW_GENERATOR

    def _generate_X(self, indexes):
        dfs = []
        for video_id in indexes:
            df_video = read_video(video_id, self.videos_folder)
            if self.interpolation_threshold is not None:
                df_video = self._apply_likelihood_filter(df_video, self.interpolation_threshold)
            df_video = df_video[self.bodyparts]
            if self.drop_likelihood:
                df_video.drop('likelihood', axis=1, level='coords')
            df_video = self._fix_video_len(df_video)
            df_video = df_video[self.cutoff:]
            dfs.append(df_video)
        X = np.stack(dfs)
        if self.scaler != None:
            try:
                self.scaler.scale_ 
                X = self._apply_scaler(X)
            except AttributeError:
                X = self._fit_apply_scaler(X)
        return X

    def _fix_video_len(self, df_video):
        """
        Fix video length to self.input_sequence_len number of frames. In case videos are
        shorter, duplicate the existing frames until the input length is reached (i.e. the video
        will be 'played' more than once).
        """
        while len(df_video) < self.input_sequence_len:
            df_video = df_video.append(df_video, ignore_index=True)
        df_video = df_video.head(self.input_sequence_len)
        return df_video

    def _fit_apply_scaler(self,X):
        cut = int(X.shape[1] / 2)
        longX = X[:, -cut:, :]
        longX = longX.reshape((longX.shape[0] * longX.shape[1], longX.shape[2]))
        flatX = X.reshape((X.shape[0] * X.shape[1], X.shape[2]))
        self.scaler.fit(longX)
        flatX = self.scaler.transform(flatX)
        return flatX.reshape((X.shape))

    def _apply_scaler(self,X):
        flatX = X.reshape((X.shape[0] * X.shape[1], X.shape[2]))
        flatX = self.scaler.transform(flatX)
        return flatX.reshape((X.shape))

    def _get_scaler(self):
        return self.scaler

    def _apply_likelihood_filter(self, df_video, likelihood):
        # when the likelihood is under the threshold, make x and y NaN
        for b in df_video.columns.get_level_values('bodyparts').unique():
            likelihood_mask = df_video[b]['likelihood'] < likelihood
            for ax in ['x', 'y']:
                df_video[b, ax][likelihood_mask] = np.NaN

        df_video = df_video.interpolate(limit_direction='both')

        # if bodypart is completely NaN, use the other side
        TWO_SIDED_BODYPARTS = ['ankle', 'knee', 'hip', 'wrist', 'elbow', 'shoulder']
        for b in TWO_SIDED_BODYPARTS:
            for ax in ['x', 'y']:
                if np.count_nonzero(~np.isnan(df_video.loc[:,(b+'1', ax)])) == 0:
                    df_video.loc[:,(b+'1', ax)] = df_video.loc[:,(b+'2', ax)]
                if np.count_nonzero(~np.isnan(df_video.loc[:,(b+'2', ax)])) == 0:
                    df_video.loc[:,(b+'2', ax)] = df_video.loc[:,(b+'1', ax)]
        return df_video

class EngineeredFeaturesDataGenerator(DataGeneratorBase):
    def __init__(self, scores_df, feature_conf: FeatureConfiguration, batch_size=1,
                 videos_folder=LYING_VIDEOS_DATA_FOLDER):
        super().__init__(scores_df, batch_size, videos_folder)
        self.feature_conf = feature_conf

    def _generate_X(self, indexes):
        results = []
        for video_id, side in indexes:
            df_video = read_video(video_id, self.videos_folder)
            features = self._get_features(df_video, side)
            results.append(features)
        return np.stack(results)

    def _get_features(self, df_video, side):
        features = list()
        features += [get_dtmp_distribution_statistics(df_video, side, bodypart)
                     for bodypart in self.feature_conf.dmtp_bodyparts]
        features += [get_dtl_distribution_statistics(df_video, side, bodypart)
                     for bodypart in self.feature_conf.dtl_bodyparts]
        features += [get_angle_distribution_statistics(df_video, side, bodyparts)
                     for bodyparts in self.feature_conf.angle_bodypart_triples]
        return np.ravel(features)
