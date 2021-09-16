from helpers import read_scores
from src.data_generators import RawDataGenerator, EngineeredFeaturesDataGenerator
from src.data_selection import MultipleScoreSelector, SplitScoreSelector


def test_data_generators():
    scores_df = read_scores()
    scores_df = MultipleScoreSelector().transform(scores_df)
    test_generator = RawDataGenerator(scores_df)
    X, y = test_generator.__getitem__(0)
    assert X.shape == (1, 501, 42)
    assert y.shape == (1, 1)


def test_engineered_feautures_data_generator():
    scores_df = read_scores()
    scores_df = SplitScoreSelector().transform(scores_df)
    test_generator = EngineeredFeaturesDataGenerator(scores_df, batch_size=2)
    X, y = test_generator.__getitem__(0)
    assert X.shape == (1, 30)
    assert y.shape == (1, 1)
