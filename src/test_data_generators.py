from src.data_generators import RawDataGenerator

def test_data_generators():
    test_generator = RawDataGenerator()
    bla = test_generator.__getitem__(0)
    print(test_generator)
