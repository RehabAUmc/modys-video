import pandas as pd
import numpy as np
import re
import os
import helpers
from statistics import get_per_part


def get_distance(p1X, p1Y, p2X, p2Y):
    ''' Returns the distance between (p1X, p1Y) and (p2X, p2Y), used for get_dtmp() '''
    return np.linalg.norm(np.array([p1X, p1Y])-np.array([p2X, p2Y]))

def get_distance_to_line(p1, p2, p3):
    ''' Returns the distance between p3 and the line between p1 and p2, used for get_dtl() '''
    return np.abs(np.cross(p2-p1,p3-p1))/np.linalg.norm(p2-p1)

def get_pol_coords(df):
    df_pol = pd.DataFrame()
    bodyparts = ['Ankle', 'Knee', 'Hip', 'Wrist', 'Elbow', 'Shoulder']
    for bodypart in bodyparts:
        df_pol['right Rho '+bodypart] = np.sqrt(df[bodypart.lower()+'1']**2 + df[bodypart.lower()+'1.1']**2)
        df_pol['right Phi '+bodypart] = np.arctan2(df[bodypart.lower()+'1.1'], df[bodypart.lower()+'1'])
        df_pol['left Rho '+bodypart] = np.sqrt(df[bodypart.lower()+'2']**2 + df[bodypart.lower()+'2.1']**2)
        df_pol['left Phi '+bodypart] = np.arctan2(df[bodypart.lower()+'2.1'], df[bodypart.lower()+'2'])
    return df_pol

def get_velocities(df, fps):
    ''' Returns a DataFrame with the velocities of all the bodyparts '''
    dist = df.diff().fillna(0.)
    bodyparts = ['Ankle', 'Knee', 'Hip', 'Wrist', 'Elbow', 'Shoulder']
    for bodypart in bodyparts:
        dist['right Speed '+bodypart] = np.sqrt(dist[bodypart.lower()+'1']**2 + dist[bodypart.lower()+'1.1']**2) * fps
        dist['left Speed '+bodypart] = np.sqrt(dist[bodypart.lower()+'2']**2 + dist[bodypart.lower()+'2.1']**2) * fps
    return dist.iloc[:,-12:]

def get_accelerations(df_velocities, fps):
    ''' Returns a DataFrame with the accelerations of all the bodyparts '''
    df_accelerations = df_velocities.diff() * fps
    df_accelerations.columns = [c.replace('Speed', 'Acceleration') for c in df_velocities.columns]
    return df_accelerations


def get_dtl(df_xy):
    ''' Calculates the distance of the knee and the ankle to the line drawn from the shoulder to the hip '''
    df_dist_to_point = pd.DataFrame()
    df_dist_to_point['Knee right dtl'] = df_xy.apply(lambda row: get_distance_to_line(np.array([row['hip1'],row['hip1.1']]), np.array([row['shoulder1'],row['shoulder1.1']]), np.array([row['knee1'],row['knee1.1']])), axis=1)
    df_dist_to_point['Knee left dtl'] = df_xy.apply(lambda row: get_distance_to_line(np.array([row['hip2'],row['hip2.1']]), np.array([row['shoulder2'],row['shoulder2.1']]), np.array([row['knee2'],row['knee2.1']])), axis=1)
    df_dist_to_point['Ankle right dtl'] = df_xy.apply(lambda row: get_distance_to_line(np.array([row['hip1'],row['hip1.1']]), np.array([row['shoulder1'],row['shoulder1.1']]), np.array([row['ankle1'],row['ankle1.1']])), axis=1)
    df_dist_to_point['Ankle left dtl'] = df_xy.apply(lambda row: get_distance_to_line(np.array([row['hip2'],row['hip2.1']]), np.array([row['shoulder2'],row['shoulder2.1']]), np.array([row['ankle2'],row['ankle2.1']])), axis=1)
    return df_dist_to_point

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

def df_transform(df, folder):
    '''Transforms the dataframe so that the index contains the patients number+side such as: '001 left'/'112 right' 
       and the columns contain the features, for example the different bodyparts.  '''
    indexes = helpers.get_indexes(folder)
    columns = list(dict.fromkeys([re.sub("right|left", "", column[4:])[1:] for column in df.columns]))
    df_T = pd.DataFrame(index=indexes, columns=columns)
    for index in indexes:
        for column in columns:
            df_T.loc[index, column] = np.array(df[index+' '+ column])
    return df_T

def get_features(folder, suffix, settings):
    '''Requires the file folder and the suffix of the files. 
       Returns the following feature dataframes: df_xys, df_dtmps, df_dtls, df_angles, df_velocities, df_accelerations, df_nans.'''
    df_xys = pd.DataFrame()
    df_pols = pd.DataFrame()
    df_dtmps = pd.DataFrame()
    df_dtls = pd.DataFrame()
    df_angles = pd.DataFrame()
    df_velocities = pd.DataFrame()
    df_accelerations = pd.DataFrame()
    df_nans = pd.DataFrame()

    pattern = re.compile(".*.csv")
    for file in os.listdir(folder):
        if pattern.match(file):
            print(file)
            df_xys_new = pd.DataFrame()
            df_pols_new = pd.DataFrame()
            df_angles_new = pd.DataFrame()
            df_dtmps_new = pd.DataFrame()
            df_dtls_new = pd.DataFrame()
            df_velocities_new = pd.DataFrame()
            df_accelerations_new = pd.DataFrame()
            df_nans_new = pd.DataFrame()

            df_data = helpers.read_xy(file[:3], settings, folder, suffix)
            df_pol = get_pol_coords(df_data)
            df_angle = get_angles(df_data)
            df_velocity = get_velocities(df_data, settings['fps'])
            df_acceleration = get_accelerations(df_velocity, settings['fps'])
            df_dtmp = get_dtmp(df_data)
            df_dtl = get_dtl(df_data)

            #RXY positions into df_xys
            bodyparts = ['Ankle', 'Knee', 'Hip', 'Wrist', 'Elbow', 'Shoulder']
            for bodypart in bodyparts:
                b1 = bodypart.lower() + '1'
                b11 = bodypart.lower() + '1.1'
                b2 = bodypart.lower() + '2'
                b21 = bodypart.lower() + '2.1'
                df_xys_new[file[:3] + ' right ' + bodypart + ' x'] = df_data[b1]
                df_xys_new[file[:3] + ' right ' + bodypart + ' y'] = df_data[b11] 
                df_xys_new[file[:3] + ' left ' + bodypart + ' x'] = df_data[b2]
                df_xys_new[file[:3] + ' left ' + bodypart + ' y'] = df_data[b21] 
            df_xys = pd.concat([df_xys,df_xys_new], axis=1)

            # Read the polar coordinates into df_pols
            for column in df_pol.columns:
                df_pols_new[file[:3] + ' ' + column] = df_pol[column]
            df_pols = pd.concat([df_pols, df_pols_new], axis=1)

            # Read the velocities into df_velocities
            for column in df_velocity.columns:
                df_velocities_new[file[:3] + ' ' + column] = df_velocity[column]
            df_velocities = pd.concat([df_velocities,df_velocities_new], axis=1)

            # Read the accelerations into df_accelerations
            for column in df_acceleration.columns:
                df_accelerations_new[file[:3] + ' ' + column] = df_acceleration[column]
            df_accelerations = pd.concat([df_accelerations,df_accelerations_new], axis=1)

            # Read the distance to middle points into df_dtmps
            for column in df_dtmp.columns:
                df_dtmps_new[file[:3] + ' ' + column] = df_dtmp[column]
            df_dtmps = pd.concat([df_dtmps, df_dtmps_new],axis=1)

            # Read the distance to line into df_dtls
            df_dtls_new[file[:3] + ' right dtls Knee'] = df_dtl['Knee right dtl']
            df_dtls_new[file[:3] + ' left dtls Knee'] = df_dtl['Knee left dtl'] 
            df_dtls_new[file[:3] + ' right dtls Ankle'] = df_dtl['Ankle right dtl']
            df_dtls_new[file[:3] + ' left dtls Ankle'] = df_dtl['Ankle left dtl'] 
            df_dtls = pd.concat([df_dtls, df_dtls_new],axis=1)

            # Read the knee angle into df_angles
            df_angles_new[file[:3] + ' right Angle Knee'] = df_angle['Angle Knee right']
            df_angles_new[file[:3] + ' left Angle Knee'] = df_angle['Angle Knee left'] 
            df_angles_new[file[:3] + ' right Angle Elbow'] = df_angle['Angle Elbow right']
            df_angles_new[file[:3] + ' left Angle Elbow'] = df_angle['Angle Elbow left'] 
            df_angles = pd.concat([df_angles,df_angles_new], axis=1)

            # Store the percs_right and percs_left into df_nans
            df_nans_new[file[:3] + ' right NaN Percentages'] = helpers.percs_right
            df_nans_new[file[:3] + ' left NaN Percentages'] = helpers.percs_left
            df_nans = pd.concat([df_nans, df_nans_new],axis=1)
    df_xys = df_transform(df_xys, folder)
    df_pols = df_transform(df_pols, folder)
    df_dtmps = df_transform(df_dtmps, folder)
    df_dtls = df_transform(df_dtls, folder)
    df_angles = df_transform(df_angles, folder)
    df_velocities = df_transform(df_velocities, folder)
    df_accelerations = df_transform(df_accelerations, folder)
    df_nans = df_transform(df_nans, folder)
    return df_xys, df_pols, df_dtmps, df_dtls, df_angles, df_velocities, df_accelerations, df_nans


def get_dtmp_distribution_feature(df_video, side, bodypart, statistic=np.nanmedian):
    """
    Get features calculated on distribution of distance of each xy position to their
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
