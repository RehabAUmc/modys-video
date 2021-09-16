class ScoreSelector:
    def __init__(self, scores_to_use=None, scorer_to_use='CO',):
        """

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

        Returns:
            pd.Dataframe with video id as index and the different score columns (in form
            'T0_DIS_D_RELP_R_tA_pscore')
        """
        df = df[df['scorer'] == self.scorer_to_use]
        df.index = df['video_id']
        return df[self.scores_to_use]

