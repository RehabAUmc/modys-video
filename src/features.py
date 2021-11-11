from functools import lru_cache

import numpy as np

from src.statistics import get_per_part


@lru_cache
def get_distance(p1X, p1Y, p2X, p2Y):
    ''' Returns the distance between (p1X, p1Y) and (p2X, p2Y), used for get_dtmp() '''
    return np.linalg.norm(np.array([p1X, p1Y])-np.array([p2X, p2Y]))


@lru_cache
def get_distance_to_line(p1, p2, p3):
    ''' Returns the distance between p3 and the line between p1 and p2, used for get_dtl() '''
    return np.abs(np.cross(p2-p1,p3-p1))/np.linalg.norm(p2-p1)


@lru_cache
def get_angle(distance1, distance2, distance3):
    ''' Returns the angle between 3 distances. '''
    return  np.arccos((np.square(distance1)+np.square(distance2)-np.square(distance3))/(2*distance1*distance2))


@lru_cache
def get_dtmp_distribution_statistics(df_video, side, bodypart, statistic=np.nanmedian):
    """
    Get statistics calculated on distribution of distance of each xy position to their
    middlepoint (i.e. average position).

    Returns:
        1 x n_features numpy array
    """
    side_id = '1' if side == 'right' else '2'
    x_mean = df_video[(bodypart + side_id, 'x')].mean()
    y_mean = df_video[(bodypart + side_id, 'y')].mean()
    distances = df_video.apply(lambda row: get_distance(x_mean, y_mean,
                                                        row[(bodypart + side_id, 'x')],
                                                        row[(bodypart + side_id, 'y')]), axis=1)
    statistics = get_per_part(distances, 10, statistic)
    return statistics


@lru_cache
def get_dtl_distribution_statistics(df_video, side, bodypart, statistic=np.nanmedian):
    """
    Get statistics calculated on distribution of distance to the line drawn from the shoulder to
    the hip.

    Returns:
        1 x n_features numpy array
    """
    side_id = '1' if side == 'right' else '2'
    distances = df_video.apply(lambda row: get_distance_to_line(
        np.array([row[('hip' + side_id, 'x')], row[('hip' + side_id, 'y')]]),
        np.array([row[('shoulder' + side_id, 'x')], row[('shoulder' + side_id, 'y')]]),
        np.array([row[(bodypart + side_id, 'x')], row[(bodypart + side_id, 'y')]])
        ), axis=1)
    statistics = get_per_part(distances, 10, statistic)
    return statistics


@lru_cache
def get_angle_distribution_statistics(df_video, side, bodyparts, statistic=np.nanmedian):
    """
    Get statistics calculated on distribution of the angle between three bodyparts.

    Returns:
        1 x n_features numpy array
    """
    side_id = '1' if side == 'right' else '2'
    angles = df_video.apply(lambda row: get_angle( get_distance(row[(bodyparts[0] + side_id, 'x')], 
                                                                row[(bodyparts[0] + side_id, 'y')],
                                                                row[(bodyparts[1] + side_id, 'x')], 
                                                                row[(bodyparts[1] + side_id, 'y')]), 
                                                   get_distance(row[(bodyparts[0] + side_id, 'x')], 
                                                                row[(bodyparts[0] + side_id, 'y')],
                                                                row[(bodyparts[2] + side_id, 'x')], 
                                                                row[(bodyparts[2] + side_id, 'y')]),
                                                   get_distance(row[(bodyparts[1] + side_id, 'x')], 
                                                                row[(bodyparts[1] + side_id, 'y')],
                                                                row[(bodyparts[2] + side_id, 'x')], 
                                                                row[(bodyparts[2] + side_id, 'y')])), axis=1)
    statistics = get_per_part(angles, 10, statistic)
    return statistics
