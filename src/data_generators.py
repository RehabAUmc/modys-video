from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from keras.utils.data_utils import Sequence

from src.features import get_dtmp_distribution_statistics, get_dtl_distribution_statistics
from src.helpers import read_video

VALID_BODYPARTS = ['ankle', 'knee', 'hip', 'wrist', 'elbow', 'shoulder', 'forehead', 'chin']


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
    def __init__(self, scores_df, batch_size=1,
                 videos_folder='../data/data_lying_052929'):
        self.batch_size = batch_size
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
    def __init__(self, scores_df, batch_size=1,
                 videos_folder='../data/data_lying_052929',
                 drop_likelihood=True,
                 input_sequence_len=501):
        super().__init__(scores_df, batch_size, videos_folder)
        self.drop_likelihood = drop_likelihood
        self.input_sequence_len = input_sequence_len

    def _generate_X(self, indexes):
        dfs = []
        for video_id in indexes:
            df_video = read_video(video_id, self.videos_folder)
            if self.drop_likelihood:
                df_video.drop('likelihood', axis=1, level='coords')
            df_video = self._fix_video_len(df_video)
            dfs.append(df_video)
        return np.stack(dfs)

    def _fix_video_len(self, df_video):
        """
        Fix video length to self.input_sequence_len number of frames. Fill with empty rows in
        case videos are shorter.
        """
        df_video = df_video.head(self.input_sequence_len)

        # Insert empty rows in case we do not have enough frames
        for _ in range(self.input_sequence_len - len(df_video)):
            df_video = df_video.append(pd.Series(), ignore_index=True)

        return df_video


class EngineeredFeaturesDataGenerator(DataGeneratorBase):
    def __init__(self, scores_df, feature_conf: FeatureConfiguration, batch_size=1,
                 videos_folder='../data/data_lying_052929'):
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
        return np.ravel(features)
