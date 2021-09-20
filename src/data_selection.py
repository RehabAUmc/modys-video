import pandas as pd


class MultipleScoreSelector:
    def __init__(self, scores_to_use=None, scorer_to_use='CO',):
        """
        Select scores from 'scores' dataframe, selecting on:
         * the score names specified
         * the scorer

        Args:
            scores_to_use: list of scores to use (i.e. ['T0_DIS_D_RLP_R_tA_pscore'])
            scorer_to_use: identifier of the scorer to use (i.e. 'CO')
        """
        if scores_to_use is None:
            self.scores_to_use = ['T0_DIS_D_RLP_R_tA_pscore']
        else:
            self.scores_to_use = scores_to_use
        self.scorer_to_use = scorer_to_use

    def transform(self, df):
        """
        Select y data

         Args:
            df: Scores pd.Dataframe with an assesment from a single annotator on a specific video on
                each row. Example:

                  video_id    ID group  T0_DIS_D_LLP_R_tA_pscore  T0_DIS_D_RLP_R_tA_pscore
              0        001  1001     A                      0.75                      0.50
              1        031  1001     A                      0.50                      0.75

        Returns:
            pd.Dataframe with video id as index and the different score columns (in form
            'T0_DIS_D_RELP_R_tA_pscore')
        """
        df = df[df['scorer'] == self.scorer_to_use]
        df.index = df['video_id']
        return df[self.scores_to_use]


class SplitScoreSelector:
    def __init__(self, left_score: str = 'T0_DIS_D_LLP_R_tA_pscore',
                 right_score: str = 'T0_DIS_D_RLP_R_tA_pscore', scorer_to_use='CO'):
        """
        Transform the 'score' dataframe into a dataframe with multiindex on video_id and side
        and the corresponding score (i.e. 'left' or 'right') as only column
        Args:
            right_score: name of score to use for right side (i.e. ['T0_DIS_D_RLP_R_tA_pscore'])
            left_score: name of score to use for left side (i.e. ['T0_DIS_D_RLP_R_tA_pscore'])
            scorer_to_use: identifier of the scorer to use (i.e. 'CO')
        """
        self.left_score = left_score
        self.right_score = right_score
        self.scorer_to_use = scorer_to_use

    def transform(self, df):
        """
        Select y data

        Args:
            df: Scores pd.Dataframe with an assesment from a single annotator on a specific video on
                each row. Example:

                  video_id    ID group  T0_DIS_D_LLP_R_tA_pscore  T0_DIS_D_RLP_R_tA_pscore
              0        001  1001     A                      0.75                      0.50
              1        031  1001     A                      0.50                      0.75

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
        df = df[[self.left_score, self.right_score]]
        df = df.rename(columns={self.left_score: "left", self.right_score: "right"})
        df = pd.DataFrame(df.stack())
        df.index = df.index.set_names(['video_id', 'side'])
        df.columns = ['score']
        return df
