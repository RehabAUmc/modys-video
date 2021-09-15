from pathlib import Path

import numpy as np
import pandas as pd

from keras.utils.data_utils import Sequence

from helpers import read_video, fix_video_id


class RawDataGenerator(Sequence):
    def __init__(self, batch_size=1, scores_to_use=None, scorer='CO',
                 videos_folder='../data/data_lying_052929', drop_likelihood=True):
        self.batch_size = batch_size
        if scores_to_use is None:
            self.scores_to_use = ['T0_DIS_D_RLP_R_tA_pscore']
        else:
            self.scores_to_use = scores_to_use
        self.scorer = scorer
        self.df_y = self._prepare_y()
        self.video_ids = self.df_y.index
        self.videos_folder = videos_folder
        if not Path(videos_folder).exists():
            raise FileNotFoundError('The path to videos folder does not exist')
        self.drop_likelihood = drop_likelihood

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.video_ids) / self.batch_size))

    def _prepare_y(self):
        df = self._read_y()
        df = self._select_y(df, self.scores_to_use, self.scorer)
        return df

    def _read_y(self):
        """
        Read y values
        TODO: Make path configurable
        """
        df = pd.read_excel('../data/data_Scoring_DIS_proximal_trunk_V1.0.xlsx')
        df['video_id'] = df['video'].apply(fix_video_id)
        df.replace(999, np.nan)
        return df

    def _select_y(self, df, scores_to_use, scorer_to_use):
        """
        Select data

        Args:
            scores_to_use: list of scores to use (i.e. ['T0_DIS_D_RLP_R_tA_pscore'])
            scorer_to_use: identifier of the scorer to use (i.e. 'CO')

        Returns:
            pd.Dataframe with video id as index and the different score columns (in form
            'T0_DIS_D_RELP_R_tA_pscore')
        """
        df = df[df['scorer'] == scorer_to_use]
        df.index = df['video_id']
        return df[scores_to_use]

    def _generate_y(self, video_ids):
        return self.df_y.iloc[video_ids].values

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
