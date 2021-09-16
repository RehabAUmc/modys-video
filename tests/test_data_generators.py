from helpers import read_scores
from src.data_generators import RawDataGenerator
from src.data_selection import ScoreSelector


def test_data_generators():
    scores_df = read_scores()
    scores_df = ScoreSelector().transform(scores_df)
    test_generator = RawDataGenerator(scores_df)
    X, y = test_generator.__getitem__(0)
    assert X.shape == (1, 501, 42)
    assert y.shape == (1, 1)
