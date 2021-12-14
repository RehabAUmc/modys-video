import pytest
from sklearn.preprocessing import StandardScaler

from src.data_generators import (RawDataGenerator, EngineeredFeaturesDataGenerator,
                                 FeatureConfiguration)
from src.data_selection import MultipleScoreSelector, SplitScoreSelector
from src.helpers import read_scores


def test_feature_configuration():
    conf = FeatureConfiguration(dmtp_bodyparts=['knee', 'forehead'])
    assert conf.dmtp_bodyparts == ['knee', 'forehead']


def test_feature_configuration_non_valid():
    with pytest.raises(AssertionError):
        _ = FeatureConfiguration(dmtp_bodyparts=['weird_bodypart'])


def test_data_generators():
    scores_df = read_scores()
    scores_df = MultipleScoreSelector().transform(scores_df)
    test_generator = RawDataGenerator(scores_df, batch_size=2,
                                      cutoff=50, interpolation_threshold=0.7,
                                      scaler=StandardScaler())
    for i in range(len(test_generator)):
        X, y = test_generator.__getitem__(i)
        assert X.shape == (2, 451, 18)
        assert y.shape == (2, 1)


def test_engineered_feautures_data_generator():
    scores_df = read_scores()
    scores_df = SplitScoreSelector().transform(scores_df)
    feature_conf = FeatureConfiguration(dmtp_bodyparts=['ankle', 'hip'],
                                        dtl_bodyparts=['ankle', 'knee'])
    test_generator = EngineeredFeaturesDataGenerator(scores_df, feature_conf, batch_size=2)
    for i in range(len(test_generator)):
        X, y = test_generator.__getitem__(i)
        assert X.shape == (2, 40)  # 2 samples in batch, 4 features with 10 bins
        assert y.shape == (2, 1)
