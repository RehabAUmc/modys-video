from pathlib import Path

import numpy as np
from keras.utils.data_utils import Sequence

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
        return self.scores_df.iloc[video_ids].values

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
