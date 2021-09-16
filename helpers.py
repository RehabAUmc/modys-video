import re
import os
from pathlib import Path

import pandas as pd
import numpy as np
import scipy.signal as signal
import features

percs_left = []
percs_right = []

def likelihoodnan(df, likeli):
    ''' Checks the .2 column from the DeepLabCut data for the likelihood. 
        If this likelihood is under the likelihood threshold the value is skipped.
        If all values from a leg are skipped the child is probably laying on their side and the values of the other leg are used. '''
    global percs_right
    global percs_left
    bodyparts = ['ankle', 'knee', 'hip', 'wrist', 'elbow', 'shoulder']
    size = len(df)
    percs_right = []
    percs_left = []
    for bodypart in bodyparts:
        b1 = bodypart + '1'
        b11 = bodypart + '1.1'
        b12 = bodypart + '1.2'
        b2 = bodypart + '2'
        b21 = bodypart + '2.1'
        b22 = bodypart + '2.2'
        df[b1] = df.apply(lambda row: np.NaN if row[b12] < likeli else row[b1], axis=1)
        df[b11] = df.apply(lambda row: np.NaN if row[b12] < likeli else row[b11], axis=1)
        df[b2] = df.apply(lambda row: np.NaN if row[b22] < likeli else row[b2], axis=1)
        df[b21] = df.apply(lambda row: np.NaN if row[b22] < likeli else row[b21], axis=1)
        # take values from other leg if there are only NaN's
        if np.count_nonzero(np.isnan(df[b1])) == size:
            df[b1] = df[b2]
        if np.count_nonzero(np.isnan(df[b11])) == size:
            df[b11] = df[b21]
        if np.count_nonzero(np.isnan(df[b2])) == size:
            df[b2] = df[b1]
        if np.count_nonzero(np.isnan(df[b21])) == size:
            df[b21] = df[b11]
        # percentages of NaN for both sides
        percs_right.append(np.count_nonzero(np.isnan(df[b1]))/size)
        percs_left.append(np.count_nonzero(np.isnan(df[b2]))/size)
    return df


def get_filter(order, lowcut, highcut, fs, filtertype):
    ''' Returns either a bandpass, lowpass or highpass filter. '''
    if filtertype == 'bp':
        return signal.butter(order, [lowcut,highcut], 'bp',fs=fs, output='sos')
    if filtertype == 'lp':
        return signal.butter(order, highcut, 'lp',fs=fs, output='sos')
    if filtertype == 'hp':
        return signal.butter(order, lowcut, 'hp',fs=fs, output='sos')

def apply_filt(df, filt):
    ''' Applies the filter on all the XY columns. '''
    bodyparts = ['ankle', 'knee', 'hip', 'wrist', 'elbow', 'shoulder']
    for bodypart in bodyparts:
        b1 = bodypart + '1'
        b11 = bodypart + '1.1'
        b2 = bodypart + '2'
        b21 = bodypart + '2.1'
        df[b1] = signal.sosfilt(filt, df[b1])
        df[b11] = signal.sosfilt(filt, df[b11])
        df[b2] = signal.sosfilt(filt, df[b2])
        df[b21] = signal.sosfilt(filt,df[b21])
    return df

def fix_video_id(video_id):
    ''' Function used by get_heights(), the indexes in the heights file don't contain the
    preceeding zeroes, so these are added. '''
    strnumber = str(video_id)
    if len(strnumber) == 1:
        return '00'+strnumber
    if len(strnumber) == 2:
        return '0'+strnumber
    return strnumber

def get_heights(file):
    ''' Read in the heights for normalization. '''
    df_heights = pd.read_excel(file)[['videonr','Height']]
    df_heights['videonr'] = df_heights.apply(lambda row: fix_video_id(row['videonr']), axis=1)
    return df_heights

def normalize(df, number, df_heights):
    ''' XY data is normalized based on the patients height. '''
    height = df_heights.loc[df_heights['videonr']==number]['Height'].values[0]
    real_TL = (-30.8 + height) / 3.26

    df['TL Right'] = df.apply(lambda row: features.get_distance(row['ankle1'], row['ankle1.1'], row['knee1'], row['knee1.1']), axis=1)
    df['TL Left'] = df.apply(lambda row: features.get_distance(row['ankle2'], row['ankle2.1'], row['knee2'], row['knee2.1']), axis=1)

    max_TL_right = np.max(df['TL Right'])
    max_TL_left = np.max(df['TL Left'])
    
    bodyparts = ['ankle', 'knee', 'hip', 'wrist', 'elbow', 'shoulder']
    for bodypart in bodyparts:
        b1 = bodypart + '1'
        b11 = bodypart + '1.1'
        b2 = bodypart + '2'
        b21 = bodypart + '2.1'
        df[b1] = df[b1]*(1/real_TL)*(1/max_TL_right)*100
        df[b11] = df[b11]*(1/real_TL)*(1/max_TL_right)*100
        df[b2] = df[b2]*(1/real_TL)*(1/max_TL_left)*100
        df[b21] = df[b21]*(1/real_TL)*(1/max_TL_left)*100
    return df


def get_filenumbers(folder):
    ''' Get the first 3 values of every csv file in the provided folder. '''
    numbers = []
    pattern = re.compile(".*.csv")
    for file in os.listdir(folder):
        if pattern.match(file):
            numbers.append(file[:3])
    return numbers

def get_indexes(folder):
    ''' Add right and left to every index. '''
    numbers = get_filenumbers(folder)
    # indexes are the numbers (001, 024, etc) + right/left
    indexes = []
    for n in numbers:
        indexes.extend([n+' right', n+' left'])
    return indexes

def read_xy(number, settings, folder, suffix):
    ''' Read the xy file using the global settings for filter, likelihood, cutoff and normalization.
    Requires:
      1) file video_id: e.g. 001
      2) settings: for reading the data
      3) folder: the data folder
      4) suffix: the file name string following the file video_id
    Returns:
      A DataFrame containing the xy data with their likelihoods.
    '''
    file = number + suffix
    headers = pd.read_csv(folder+file, header = 1, nrows = 1)
    df = pd.read_csv(folder+file,header=2, names = headers.columns)
    if settings['likeli_on']:
        df = likelihoodnan(df, settings['likeli'])
        df = df.interpolate(limit_direction='both')
    if settings['filt_on']:
        df = apply_filt(df, settings['filt'])
    if settings['normalize_on']:
        df_heights = get_heights(settings['heights_file'])
        df = normalize(df, number, df_heights)
    if settings['cutoff_on']:
        df = df[settings['cutoff']:]
    return df


def read_video(video_id, videos_folder='../data/data_lying_052929'):
    """

    Args:
        video_id: id of the video (i.e. '001')
        videos_folder: folder of the videos (i.e. 'data/data_lying_052929)

    Returns:
        pd.Dataframe of shape n_frames x n_bodyparts. Columns are multiindexed with bodyparts,
        coords. For example, to access all 'x' coordinates do df.xs('x', axis=1, level='coords').
    """
    paths = [path for path in Path(videos_folder).glob('*.csv')
             if path.name.startswith(video_id)]
    if len(paths) == 0:
        raise FileNotFoundError(f'Video with video_id: {video_id} not found')
    if len(paths) > 1:
        raise RuntimeError(f'More than 1 video with video_id {video_id} found')
    filepath = paths[0]
    df = pd.read_csv(filepath, index_col=0, header=[1, 2])
    return df


def read_scores(filepath: str = '../data/data_Scoring_DIS_proximal_trunk_V1.0.xlsx'):
    """
    Read scores (y)

    Returns:
        df: Scores pd.Dataframe with an assesment from a single annotator on a specific video on
            each row. Example:
                  video_id    ID group  T0_DIS_D_LLP_R_tA_pscore  T0_DIS_D_RLP_R_tA_pscore
              0        001  1001     A                      0.75                      0.50
              1        031  1001     A                      0.50                      0.75

    """
    df = pd.read_excel(filepath)
    df['video_id'] = df['video'].apply(fix_video_id)
    df.drop(columns=['video'])
    df = df.replace(999, np.nan)
    return df
