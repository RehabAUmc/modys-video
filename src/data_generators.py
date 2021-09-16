from pathlib import Path

import numpy as np
from keras.utils.data_utils import Sequence

from features import get_dtmp_distribution_feature
from helpers import read_video


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
                 videos_folder='../data/data_lying_052929', drop_likelihood=True):
        super().__init__(scores_df, batch_size, videos_folder)
        self.drop_likelihood = drop_likelihood

    def _generate_X(self, indexes):
        dfs = []
        for video_id in indexes:
            df_video = read_video(video_id, self.videos_folder)
            if self.drop_likelihood:
                df_video.drop('likelihood', axis=1, level='coords')
            dfs.append(df_video)
        return np.stack(dfs)


class EngineeredFeaturesDataGenerator(DataGeneratorBase):
    def __init__(self, scores_df, batch_size=1,
                 videos_folder='../data/data_lying_052929', bodyparts=None):
        super().__init__(scores_df, batch_size, videos_folder)

        # features:
        self.bodyparts = bodyparts if bodyparts is not None else ['ankle', 'knee', 'hip']

    def _generate_X(self, indexes):
        results = []
        for video_id, side in indexes:
            df_video = read_video(video_id, self.videos_folder)
            features = np.ravel([get_dtmp_distribution_feature(df_video, side, bodypart)
                                 for bodypart in self.bodyparts])
            results.append(features)
        return np.stack(results)

