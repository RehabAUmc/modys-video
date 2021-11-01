from src.data_selection import SplitScoreSelector

from src.helpers import read_scores


def test_split_data_selector():
    selector = SplitScoreSelector()
    df = read_scores()
    df = selector.transform(df)
    assert len(df) == 124
    assert df.index.names == ['video_id', 'side']
    assert list(df.columns) == ['score', 'ID']
