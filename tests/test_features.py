import numpy as np

from src.features import get_dtmp_distribution_statistics, get_dtl_distribution_statistics
from src.helpers import read_video


def test_get_dtmp_distribution_statistics():
    df_video = read_video(video_id='001')
    feature = get_dtmp_distribution_statistics(df_video, 'left', 'ankle', np.nanmedian)
    assert len(feature) == 10


def test_get_dtl_distribution_statistics():
    df_video = read_video(video_id='001')
    feature = get_dtl_distribution_statistics(df_video, 'left', 'ankle', np.nanmedian)
    assert len(feature) == 10
