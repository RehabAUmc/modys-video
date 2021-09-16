import numpy as np
import pandas as pd

from statistics import get_per_part


def get_distance(p1X, p1Y, p2X, p2Y):
    ''' Returns the distance between (p1X, p1Y) and (p2X, p2Y), used for get_dtmp() '''
    return np.linalg.norm(np.array([p1X, p1Y])-np.array([p2X, p2Y]))

def get_distance_to_line(p1, p2, p3):
    ''' Returns the distance between p3 and the line between p1 and p2, used for get_dtl() '''
    return np.abs(np.cross(p2-p1,p3-p1))/np.linalg.norm(p2-p1)

def get_angles(df_xy):
    ''' Returns the angle of the knee '''
    df_distance = pd.DataFrame()
    # calculate angle right knee
    df_distance['Hip_knee right'] = np.sqrt(np.square(df_xy['knee1'] - df_xy['hip1'])+np.square(df_xy['knee1.1'] - df_xy['hip1.1']))
    df_distance['Hip_ankle right'] = np.sqrt(np.square(df_xy['ankle1'] - df_xy['hip1'])+np.square(df_xy['ankle1.1'] - df_xy['hip1.1']))
    df_distance['Knee_ankle right'] = np.sqrt(np.square(df_xy['ankle1'] - df_xy['knee1'])+np.square(df_xy['ankle1.1'] - df_xy['knee1.1']))
    df_distance['Angle Knee right'] = np.arccos((np.square(df_distance['Hip_knee right'])+np.square(df_distance['Hip_ankle right'])-np.square(df_distance['Knee_ankle right']))/(2*df_distance['Hip_knee right']*df_distance['Hip_ankle right']))
    
    # calculate angle left knee
    df_distance['Hip_knee left'] = np.sqrt(np.square(df_xy['knee2'] - df_xy['hip2'])+np.square(df_xy['knee2.1'] - df_xy['hip2.1']))
    df_distance['Hip_ankle left'] = np.sqrt(np.square(df_xy['ankle2'] - df_xy['hip2'])+np.square(df_xy['ankle2.1'] - df_xy['hip2.1']))
    df_distance['Knee_ankle left'] = np.sqrt(np.square(df_xy['ankle2'] - df_xy['knee2'])+np.square(df_xy['ankle2.1'] - df_xy['knee2.1']))
    df_distance['Angle Knee left'] = np.arccos((np.square(df_distance['Hip_knee left'])+np.square(df_distance['Hip_ankle left'])-np.square(df_distance['Knee_ankle left']))/(2*df_distance['Hip_knee left']*df_distance['Hip_ankle left']))
    
    # calculate angle right elbow
    df_distance['Shoulder_elbow right'] = np.sqrt(np.square(df_xy['elbow1'] - df_xy['shoulder1'])+np.square(df_xy['elbow1.1'] - df_xy['shoulder1.1']))
    df_distance['Shoulder_wrist right'] = np.sqrt(np.square(df_xy['wrist1'] - df_xy['shoulder1'])+np.square(df_xy['wrist1.1'] - df_xy['shoulder1.1']))
    df_distance['Elbow_wrist right'] = np.sqrt(np.square(df_xy['wrist1'] - df_xy['elbow1'])+np.square(df_xy['wrist1.1'] - df_xy['elbow1.1']))
    df_distance['Angle Elbow right'] = np.arccos((np.square(df_distance['Shoulder_elbow right'])+np.square(df_distance['Shoulder_wrist right'])-np.square(df_distance['Elbow_wrist right']))/(2*df_distance['Shoulder_elbow right']*df_distance['Shoulder_wrist right']))
    
    # calculate angle left elbow
    df_distance['Shoulder_elbow left'] = np.sqrt(np.square(df_xy['elbow2'] - df_xy['shoulder2'])+np.square(df_xy['elbow2.1'] - df_xy['shoulder2.1']))
    df_distance['Shoulder_wrist left'] = np.sqrt(np.square(df_xy['wrist2'] - df_xy['shoulder2'])+np.square(df_xy['wrist2.1'] - df_xy['shoulder2.1']))
    df_distance['Elbow_wrist left'] = np.sqrt(np.square(df_xy['wrist2'] - df_xy['elbow2'])+np.square(df_xy['wrist2.1'] - df_xy['elbow2.1']))
    df_distance['Angle Elbow left'] = np.arccos((np.square(df_distance['Shoulder_elbow left'])+np.square(df_distance['Shoulder_wrist left'])-np.square(df_distance['Elbow_wrist left']))/(2*df_distance['Shoulder_elbow left']*df_distance['Shoulder_wrist left']))
    return df_distance


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
