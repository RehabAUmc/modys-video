import numpy as np
import pandas as pd

from keras.utils.data_utils import Sequence

import helpers


class RawDataGenerator(Sequence):
    def __init__(self, batch_size=1, scores_to_use=None, scorer='CO'):
        self.batch_size = batch_size
        if scores_to_use is None:
            self.scores_to_use = ['T0_DIS_D_RLP_R_tA_pscore']
        else:
            self.scores_to_use = scores_to_use
        self.scorer = scorer
        self.df_y = self._prepare_y()
        self.indexes = self.df_y.index

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.indexes) / self.batch_size))

    def _prepare_y(self):
        df = self._read_y()
        df = self._select_y(df, self.scores_to_use, self.scorer)
        return df

    def _read_y(self):
        """
        Read y values
        TODO: Fix path?
        """
        df = pd.read_excel('../data/data_Scoring_DIS_proximal_trunk_V1.0.xlsx')
        df['video_id'] = df['video'].apply(helpers.fix_video_id)
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

    def _generate_y(self, indexes):
        return self.df_y.iloc[indexes].values

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        # Generate data
        X = None
        y = self._generate_y(indexes)
        return X, y
