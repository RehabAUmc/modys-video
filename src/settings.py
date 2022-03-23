from pathlib import Path

PROJECT_ROOT_FOLDER = Path(__file__).parent.parent
DATA_FOLDER = PROJECT_ROOT_FOLDER / 'data'
SCORES_DATA_FILEPATH = DATA_FOLDER / 'data_clinical_scoring.xlsx'
LYING_VIDEOS_DATA_FOLDER = DATA_FOLDER / 'data_stickfigure_coordinates_lying'
SITTING_VIDEOS_DATA_FOLDER = DATA_FOLDER / 'data_stickfigure_coordinates_sitting'
