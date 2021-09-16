from pathlib import Path

import numpy as np
from keras.utils.data_utils import Sequence

from features import get_dtmp_distribution_feature
from helpers import read_video


class RawDataGenerator(Sequence):
    def __init__(self, scores_df, batch_size=1,
                 videos_folder='../data/data_lying_052929', drop_likelihood=True):
        self.batch_size = batch_size
        self.scores_df = scores_df
        self.video_ids = self.scores_df.index
        self.videos_folder = videos_folder
        if not Path(videos_folder).exists():
            raise FileNotFoundError('The path to videos folder does not exist')
        self.drop_likelihood = drop_likelihood

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.video_ids) / self.batch_size))

    def _generate_y(self, video_ids):
        return self.scores_df.loc[video_ids].values

    def _generate_X(self, video_ids):
        dfs = []
        for video_id in video_ids:
            df_video = read_video(video_id, self.videos_folder)
            if self.drop_likelihood:
                df_video.drop('likelihood', axis=1, level='coords')
            dfs.append(df_video)
        return np.stack(dfs)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        video_ids = self.video_ids[index * self.batch_size: (index + 1) * self.batch_size]

        # Generate data
        X = self._generate_X(video_ids)
        y = self._generate_y(video_ids)
        return X, y


class EngineeredFeaturesDataGenerator(Sequence):
    def __init__(self, scores_df, batch_size=1,
                 videos_folder='../data/data_lying_052929', bodyparts=None):
        self.batch_size = batch_size
        self.scores_df = scores_df
        self.indexes = self.scores_df.index
        self.videos_folder = videos_folder
        if not Path(videos_folder).exists():
            raise FileNotFoundError('The path to videos folder does not exist')

        # features:
        self.bodyparts = bodyparts if bodyparts is not None else ['ankle', 'knee', 'hip']

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.indexes) / self.batch_size))

    def _generate_y(self, indexes):
        return self.scores_df.loc[indexes].values

    def _generate_X(self, indexes):
        results = []
        for video_id, side in indexes:
            df_video = read_video(video_id, self.videos_folder)
            features = np.ravel([get_dtmp_distribution_feature(df_video, side, bodypart)
                                 for bodypart in self.bodyparts])
            results.append(features)
        return np.stack(results)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        # Generate data
        X = self._generate_X(indexes)
        y = self._generate_y(indexes)
        return X, y
