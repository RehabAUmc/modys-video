from helpers import read_scores


def test_read_scores():
    df = read_scores()
    assert 'video_id' in df
    assert len(df.iloc[0]['video_id']) == 3
