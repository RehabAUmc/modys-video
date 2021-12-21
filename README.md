# MODYS: Video analysis
This project contains:
*	1 Python Jupyter Notebook: main.ipynb
*	6 Python code files:
    *	helpers.py
    *	statistics.py
    *	features.py
    *	ai_func.py
    *	ai_models.py
    *	plotting.py
    
## Reading in data
* Download all data in https://surfdrive.surf.nl/files/index.php/s/9xtLZ074DzmUB5Z to
the /data folder in this repository. 
* Unzip the .zip files.
* You should have a `data/data_lying_XXXXXX` and `data/data_sitting_XXXXXX` folder with `.csv` files, 
and a file called `data/data_Scoring_DIS_proximal_trunk_VX.X.xlsx`

## main.ipynb
main.ipynb has the following chapters:
*	Imports
*	Settings
*	Get file numbers
*	Read and store the features
  *	Get distributions
  *	Get statistics
  *	Get statistics for multiple parts
*	Get the scores
*	AI
  *	Standard algorithms
    *	Get data 
    *	Standard Linear Regression
    *	Standard Random Forrest Regressor
    *	Leave-One-Out cross-validation: multiple repeats
  *	1D CNN
    *	Get data 
    *	Simple Version
    *	Complex Version
    *	Run multiple times, with shuffled data set
  *	1D CNN with two outputs: DYS+CA
    *	Get data
    *	Simple CNN double output
    *	Run multiple times, with shuffled data set
  *	LSTM
    *	Simple LSTM
    *	Complex LSTM
## 6 Python Code Files
*	**helpers.py**
This file contains helper functions for reading in data, filtering, normalizing, etc. 
*	**statistics.py**
Functions for applying the different statistics.
*	**features.py**
Contains functions calculating features and one function reading in all the files and calculating all the features for each file.
*	**ai_func.py**
Functions for running the AI such as: scaling data, splitting data, running experiments, etc.
*	**ai_models.py**
Contains multiple Keras models.
*	**plotting.py**
Functions to create a stick-figure-movie from the XY-data. 

## Steps
The steps to run main.ipynb.
### Step 1: Imports
*	Put your required libraries in Imports
### Step 2: Settings
*	Put the right folder and file naming convention in the first variables.
*	Set the settings: do you want filtering, only values with a certain likelihood, a cutoff, normalization?
### Step 3: Get file numbers
*	The first 3 characters of the files contain the video number
*	The scores are given for the left side and right side, so indexes are the numbers +lef/right
### Step 4: Read and store the features
*	In the features.py file you can add features to the get_features() function.
*	From these features you can calculate distributions and statistics.
### Step 5: Get the scores
*	Read in the scores that the model must predict. 
*	The rows should contain the indexes, while the columns are the different scores.
### Step 6: AI
*	You can choose which features you want in the training data and which score you like to predict under the “Get data” sections. Just give the dataframe + column of the desired input data.
*	There are two sections here: the standard algorithms imported from scikit-learn and the models made in Keras. 
*	These can be run in two ways: run_plot_standard/neural() to get the scores of the model and a plot. Or run_standard/neural_experiment() to run the chosen model multiple times. You provide these functions with the data, a splitting function, the settings, and the model.

## Citation
If you want to cite this software, please use the metadata in [CITATION.cff](CITATION.cff)