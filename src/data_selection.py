import warnings

import pandas as pd

from src.helpers import read_video
from src.settings import LYING_VIDEOS_DATA_FOLDER


class ScoreSelectorBase:
    def __init__(self, videos_folder):
        self.videos_folder = videos_folder

    def _drop_missing_video_data(self, scores_df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop rows in the data if the video is missing.
        """
        missing_ids = []
        for video_id in scores_df.index:
            try:
                read_video(video_id, self.videos_folder)
            except FileNotFoundError:
                missing_ids.append(video_id)
        if missing_ids:
            warnings.warn(f'Dropping rows {missing_ids} with missing video')
            scores_df = scores_df.drop(labels=missing_ids)
        return scores_df

    def _drop_nan_scored_data(self, scores_df: pd.DataFrame, scores_to_use) -> pd.DataFrame:
        """
        Drop rows in the data if the score is missing.
        """
        scores_df_na_free = scores_df.dropna(subset=scores_to_use)
        only_na = scores_df[~scores_df.index.isin(scores_df_na_free.index)]
        warnings.warn(f'Dropping rows {only_na.index.values} with missing score')
        return scores_df_na_free
    
    def transform(self, scores_df):
        raise NotImplementedError


class MultipleScoreSelector(ScoreSelectorBase):
    def __init__(self, videos_folder=LYING_VIDEOS_DATA_FOLDER,
                 scores_to_use=None, scorer_to_use: int = 1, ):
        """
        Select scores from 'scores' dataframe, selecting on:
         * the score names specified
         * the scorer

        Args:
            scores_to_use: list of scores to use (i.e. ['D_LLP_R_tA_pscore'])
            scorer_to_use: identifier of the scorer to use (i.e. 1)
        """
        super().__init__(videos_folder)
        if scores_to_use is None:
            self.scores_to_use = ['D_LLP_R_tA_pscore']
        else:
            self.scores_to_use = scores_to_use
        self.scorer_to_use = scorer_to_use

    def transform(self, df):
        """
        Select y data

         Args:
            df: Scores pd.Dataframe with an assesment from a single annotator on a specific video on
                each row. Example:

                  video_id    ID group  D_LLP_R_tA_pscore  D_RLP_R_tA_pscoree
              0        001  1001     A  0.75               0.50
              1        031  1001     A  0.50               0.75
            videos_folder: Path to the videos folder

        Returns:
            pd.Dataframe with video id as index and the different score columns (in form
            'D_LLP_R_tA_pscore'), also has ID column so we can group based on subject id.
        """
        df = df[df['scorer'] == self.scorer_to_use]
        df.index = df['video_id']
        df = df[self.scores_to_use + ['ID']]

        df = self._drop_missing_video_data(df)
        df = self._drop_nan_scored_data(df, self.scores_to_use)
        return df


class SplitScoreSelector(ScoreSelectorBase):
    def __init__(self, videos_folder=LYING_VIDEOS_DATA_FOLDER,
                 left_score: str = 'D_LLP_R_tA_pscore',
                 right_score: str = 'D_RLP_R_tA_pscore', scorer_to_use=1):
        """
        Transform the 'score' dataframe into a dataframe with multiindex on video_id and side
        and the corresponding score (i.e. 'left' or 'right') as only column
        Args:
            right_score: name of score to use for right side (i.e. ['D_LLP_R_tA_pscore'])
            left_score: name of score to use for left side (i.e. ['D_LLP_R_tA_pscoree'])
            scorer_to_use: identifier of the scorer to use (i.e. 1)
        """
        super().__init__(videos_folder)
        self.left_score = left_score
        self.right_score = right_score
        self.scorer_to_use = scorer_to_use

    def transform(self, df):
        """
        Select y data

        Args:
            df: Scores pd.Dataframe with an assesment from a single annotator on a specific video on
                each row. Example:

                  video_id    ID group  D_LLP_R_tA_pscore  D_RLP_R_tA_pscoree
              0        001  1001     A  0.75               0.50
              1        031  1001     A  0.50               0.75

        Returns:
            pd.Dataframe with multiindex on video_id and side and the corresponding score as only
            column

                             score
            video_id side
            001      left     0.75
                     right    0.75
            031      left     0.50
                     right    0.50
        """
        df = df[df['scorer'] == self.scorer_to_use]
        df.index = df['video_id']
        subject_ids = df['ID']
        df = df[[self.left_score, self.right_score]]
        df = df.rename(columns={self.left_score: "left", self.right_score: "right"})
        df = pd.DataFrame(df.stack())
        df.index = df.index.set_names(['video_id', 'side'])
        df.columns = ['score']
        # Add the ID column again, joining on 'video_id'
        df = df.join(subject_ids)
        df = self._drop_missing_video_data(df)
        df = self._drop_nan_scored_data(df, ['score'])
        return df
