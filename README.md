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
Generate data in the right input format for the experiments; 
*	**data_selection.py:**
Read the clinical scores and match it to x,y-coordinates; drops if there is no match

## Notebooks (data generation)
* **generator_based_pipeline.ipynb:**
Basic data pipeline example

## Notebooks (machine learning experiments)
 * **experiment-2-different-random-forrest-models.ipynb:**
 Notebook trains random forest regressor models for different clinical outputs (dystonia: arm amplidute, arm duration, leg amplidute, leg duration)
 * **1.1-model-exploration-deep-learning.ipynb:**
 Notebook exploring deep learning to predict dystonia
 * **3.1-basic-deeplearning-lying.ipynb:**
 Notebook first step towards deeplearning 

## Notebooks (visualisation)
* **scoring-visualisation.ipynb:**
 Notebook visualize the scoring of the different scorers e.g. scorer 1 vs scorer 2; scorer 1 vs scorer 3
* **feature-visualisation.ipynb:**
 Within this notebook single files (stick figure data) can be visualized 
* **experiment-2-results-visualisation.ipynb:**
 Notebook includes different options for visualisation for results of experiment 2 i.e. predicted versus true scores

   
## Results folder
Results of experiments are stored as xls in the result folder

## Test folder
Automatic test for code changes

## Steps to get started
The steps to run MODYS-video
### Step 1: Data
*	Put your data in the data folder. Example data can be downloaded from https://doi.org/10.5281/zenodo.5638470 see reading in data. The notebooks can read data structured in the same way as the example data. The first 3 characters of the files contain the video number.
### Step 2: Settings
*	Put the right folder. For the example data you do not have change anything.
### Step 3 (optional) Feature - ruwe data visualisation
*  Run the feature-visualisation.ipynb for selected videos, to check data
### Step 4 Run the notebooks
*  Run the machine learning experiments notebooks
*	 Note: The code now uses coordinates of hip, knee and ankle for the lying videos and shoulder, elbow and wrist for the sitting video. This can adaped.
### Step 5: Check results folder
*	Results of predictions are stored in the result folder: scorer, true value and predicted values by model
### Step 6: Visualisation results
*	run the experiment-2-results-visualisation.ipynb. Exmaple plots are given to show the results for expmeriment 2 (random forest regressor)

## Citation
If you want to cite this software, please use the metadata in [CITATION.cff](CITATION.cff)
