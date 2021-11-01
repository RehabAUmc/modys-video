from src.ai_func import cross_validation_generator
from src.data_selection import SplitScoreSelector
from src.helpers import read_scores


def test_cross_validation():
    scores_df = read_scores()
    scores_df = SplitScoreSelector().transform(scores_df)
    n_samples = len(scores_df)
    generator = cross_validation_generator(scores_df)
    for train_scores_df, test_scores_df in generator:
        # We expect the length of train set to be around 80% and test set around 20%.
        # But because of our grouped split it can vary a bit.
        assert n_samples * 0.75 <= len(train_scores_df) <= n_samples * 0.85
        assert n_samples * 0.15 <= len(test_scores_df) <= n_samples * 0.25

