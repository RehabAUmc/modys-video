from src.data_generators import RawDataGenerator


def test_data_generators():
    test_generator = RawDataGenerator()
    X, y = test_generator.__getitem__(0)
    assert X.shape == (1, 501, 42)
    assert y.shape == (1, 1)
