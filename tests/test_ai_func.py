from src.ai_func import cross_validation_generator
from src.helpers import read_scores


def test_cross_validation():
    scores_df = read_scores()
    generator = cross_validation_generator(scores_df)
    for train_scores_df, test_scores_df in generator:
        # The scores_df has 161 samples, so we expect the length of train set to be around 80%,
        # and test set around 20%. But because of our grouped split it can vary a bit.
        assert 125 <= len(train_scores_df) <= 135
        assert 25 <= len(test_scores_df) <= 35

