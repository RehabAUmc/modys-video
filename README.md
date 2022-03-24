[![DOI](https://zenodo.org/badge/406670176.svg)](https://zenodo.org/badge/latestdoi/406670176)

# MODYS: Video analysis

MODYS-video provides a machine learning pipeline to train models to automatically assess dystonia and choreoathetosis of children with dyskinetic cerebral palsy using 2D coordinates of body points extracted from videos. The body coordinates are extraced using DeepLabcut (https://github.com/DeepLabCut/DeepLabCut). The clinical scoring is performed with the Dyskinesia Impairment Scale. Example data are available on zenodo.

![afbeelding](https://user-images.githubusercontent.com/54277291/159797915-6d0b671d-3ae0-4571-9544-891fd0fb1579.png)

## Folder structure
* data
* notebooks
* results
* scr
* tests
 
## Reading in data
* Download all data from https://doi.org/10.5281/zenodo.5638470 to
the /data folder in this repository. 
* Unzip the .zip files.
* You should have a `data/data_stickfigure_coordinates_lying` and `data/data_stickfigure_coordinates_sitting` folder with `.csv` files, 
and a file called `data/data_clinical_scoring.xlsx`

## Details Python Code Files (scr)
*	**setting.py**
This file contains the settings for the data
*	**helpers.py:**
This file contains helper functions for reading in data, filtering, normalizing, etc. 
*	**statistics.py:**
Functions for applying the different statistics.
*	**features.py:**
Contains functions calculating features and one function reading in all the files and calculating all the features for each file.
*	**ai_func.py:**
Functions for running the AI such as: scaling data, splitting data, running experiments, etc.
*	**ai_models.py:**
Contains multiple Keras models.
*	**plotting.py:**
Functions to create a stick-figure-movie from the XY-data. 
*	**data_generators.py:**
add
*	**data_selection.py:**
add

## Notebooks (data generation)
* **generator_based_pipeline.ipynb:**

## Notebooks (machine learning)
 * **experiment-2-different-random-forrest-models.ipynb:**
 Notebook trains random forest regressor models for different clinical outputs (dystonia: arm amplidute, arm duration, leg amplidute, leg duration)
 * **1.1-model-exploration-deep-learning.ipynb:**
 Notebook exploring deep learning to predict dystonia
 * **3.1-basic-deeplearning-lying.ipynb:**
 Notebook first step towards deeplearning 
  
## Results
Results of experiments are stored as xls in the result folder

## Notebooks (visualisation)
* **experiment-2-results-visualisation.ipynb:**
 Notebook includes different options for visualisation for results of experiment 2 i.e. predicted versus true scores
* **feature-visualisation.ipynb:**
 Within this notebook single files (stick figure data) can be visualized 
   
## main.ipynb (adapt)
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

## Steps (adapt)
The steps to run MODYS-video
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
