import numpy as np

from features import get_dtmp_distribution_feature
from helpers import read_video


def test_get_dmp_distribution_feature():
    df_video = read_video(video_id='001')
    feature = get_dtmp_distribution_feature(df_video, 'left', 'ankle', np.nanmedian)
    assert len(feature) == 10
