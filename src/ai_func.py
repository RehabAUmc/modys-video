import random
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

###########################
##### Display Results #####
###########################
def summarize_results(scores):
    ''' Shows the mean + std of the scores. '''
    print(scores)
    m, s = np.mean(scores), np.std(scores)
    print('MAE: %.3f (+/-%.3f)' % (m, s))

def plot_output(x, y, title, xlabel, ylabel, labels=[]):
    ''' Standard scatter plot. '''
    plt.figure(figsize=(20,20))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(0,1)
    plt.scatter(x, y)
    if labels != []:
        for i, txt in enumerate(labels):
            plt.annotate(txt, (x[i], y[i]), fontsize=16)

###########################
##### Data Functions ######
###########################
def scale_data(trainX, testX):
    ''' A scale function shamelessly stolen from: 
        https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/ '''
    # remove overlap
    cut = int(trainX.shape[1] / 2)
    longX = trainX[:, -cut:, :]
    # flatten windows
    longX = longX.reshape((longX.shape[0] * longX.shape[1], longX.shape[2]))
    # flatten train and test
    flatTrainX = trainX.reshape((trainX.shape[0] * trainX.shape[1], trainX.shape[2]))
    flatTestX = testX.reshape((testX.shape[0] * testX.shape[1], testX.shape[2]))
    s = StandardScaler()
    # fit on training data
    s.fit(longX)
    # apply to training and test data
    longX = s.transform(longX)
    flatTrainX = s.transform(flatTrainX)
    flatTestX = s.transform(flatTestX)
    # reshape
    flatTrainX = flatTrainX.reshape((trainX.shape))
    flatTestX = flatTestX.reshape((testX.shape))
    return flatTrainX, flatTestX


def cross_validation_generator(scores_df):
    """
    Args:
        scores_df: Scores pd.Dataframe with an assesment from a single annotator on a specific video on
            each row. For cross validation use output of a ScoreSelector class (See
            data_selection.py)

    Divide all the samples in 5 folds of roughly equal sizes (if possible). The prediction
    function is learned using 4 folds, and the fold left out is used for test.

    Yields:
        train_scores_df, test_scores_df tuples, corresponding to the subset of the scores_df that
        belongs to the train folds or the test folds respectively.
    """
    group_kfold = GroupKFold(n_splits=5)
    groups = scores_df['ID']
    scores_df = scores_df.drop(columns='ID')
    for train_indices, test_indices in group_kfold.split(scores_df, groups=groups):
        train_scores_df = scores_df.iloc[train_indices]
        test_scores_df = scores_df.iloc[test_indices]
        yield train_scores_df, test_scores_df


def LOO_single_output(data, ntrials, test_index):
    ''' Splits the data by the leaving one out method. Both legs from the same subject are always put in the same dataset. '''
    trainX = []
    trainy = []
    testX = []
    testy = []
    variables = len(data[0])-1
    train_indexes = list(range(0,ntrials*2,2))
    train_indexes.remove(test_index)
    for i in train_indexes:
        trainX.append(data[i][0:variables])
        trainX.append(data[i+1][0:variables])
        trainy.append(data[i][variables])
        trainy.append(data[i+1][variables])
    testX.append(data[test_index][0:variables])
    testX.append(data[test_index+1][0:variables])
    testy.append(data[test_index][variables])
    testy.append(data[test_index+1][variables])
    trainX = np.array([np.transpose(e) for e in trainX])
    trainy = np.asarray(trainy)
    testX = np.array([np.transpose(e) for e in testX])
    testy = np.asarray(testy)
    return trainX, trainy, testX, testy 

def split_data(data, ntrials, noutputs, split=0.7):
    ''' Splits the data set in 70%/30% train/test sets. Both legs from the same subject are always put in the same dataset. 
        Used for datasets with two y variables, for example DYS lower + CA lower. '''
    variables = len(data[0])-noutputs
    ntraining = math.floor(ntrials*split)
    indexes = list(range(0,ntrials*2,2)) #all even rows, in total there are 188 rows (96 patients, left+right)
    trainsample = random.sample(range(0,ntrials*2,2), ntraining) #get 70% of the rows (65 patients)
    testsample = list(set(indexes).difference(trainsample)) #get the 30% rows that were not selected for training
    trainX = []
    trainy = []
    testX = []
    testy = []
    for i in trainsample:
        trainX.append(data[i][0:variables])
        trainX.append(data[i+1][0:variables])
        if noutputs == 1:
            trainy.append(data[i][variables])
            trainy.append(data[i+1][variables])
        if noutputs == 2:
            trainy.append([data[i][variables],data[i][variables+1]])
            trainy.append([data[i+1][variables],data[i+1][variables+1]])
    for i in testsample:
        testX.append(data[i][0:variables])
        testX.append(data[i+1][0:variables])
        if noutputs == 1:
            testy.append(data[i][variables])
            testy.append(data[i+1][variables])
        if noutputs == 2:
            testy.append([data[i][variables],data[i][variables+1]])
            testy.append([data[i+1][variables],data[i+1][variables+1]])
    trainX = np.array([np.transpose(e) for e in trainX])
    trainy = np.asarray(trainy)
    testX = np.array([np.transpose(e) for e in testX])
    testy = np.asarray(testy)
    return trainX, trainy, testX, testy, trainsample, testsample

def create_balanced_batch(trainX, trainy, amount, zero_amount):
    ''' NOT IMPLEMENTED YET
    Might be useful for the neural nets to counter the high amount of zeros in the CA lower scores. 
    It shuffles the indexes and then adds the datasamples to the batch. The samples with y = 0 are only added when there are 
    not too many in the batch already. 
    Requires: 
      trainX: the train X data
      trainy: the train y data
      amount: amount of samples in the batch
      zero_amount: amount of samples allowed to be zero
    Returns batchX and batchy '''
    count = 0
    zero_count = 0
    batchX = []
    batchy = []
    batchsample = list(range(0,len(trainX),2))
    random.shuffle(batchsample)
    i = 0 
    while count < amount:
        index = batchsample[i]
        X1 = trainX[index]
        X2 = trainX[index+1]
        y1 = trainy[index]
        y2 = trainy[index+1]
        if y1 == 0:
            if zero_count < zero_amount:
                count += 1
                zero_count += 1
                batchX.append(X1)
                batchy.append(y1)
        else:
            count += 1
            batchX.append(X1)
            batchy.append(y1)
        if y2 == 0:
            if zero_count < zero_amount:
                batchX.append(X2)
                batchy.append(y2)
                count += 1
                zero_count += 1
        else:
            batchX.append(X2)
            batchy.append(y2)
            count += 1
        i += 1
    return batchX, batchy

########################
##### Plot and run #####
########################

def run_standard_experiment(data, repeats, ntrials, noutputs, model):
    ''' Runs a given standard model N times. '''
    scores = list()
    predys = list()
    testys = list()
    for _ in range(repeats):
        trainX, trainy, testX, testy, _, _ = split_data(data, ntrials, noutputs)
        model.fit(trainX, trainy)
        predy = model.predict(testX)
        score = mean_absolute_error(testy, predy)
        scores.append(score)    
        predys.extend(predy)
        testys.extend(testy)
    summarize_results(scores)
    return testys, predys

def run_standard_LOO_experiment(data, ntrials, model):
    ''' Experiment using Leave-One-Out. '''
    scores = list()
    predys = list()
    testys = list()
    for i in range(0,ntrials*2,2):
        trainX, trainy, testX, testy = LOO_single_output(data, ntrials, i)
        model.fit(trainX, trainy)
        predy = model.predict(testX)
        score = mean_absolute_error(testy, predy)
        scores.append(score)    
        predys.extend(predy)
        testys.extend(testy)
    summarize_results(scores)
    return testys, predys

def run_neural_experiment(data, repeats, ntrials, noutputs, model, ai_settings):
    ''' Runs a given neural model N times. '''
    scores = list()
    hists = list()
    outps = list()
    for r in range(repeats):
        trainX, trainy, testX, testy, _, _ = split_data(data, ntrials, noutputs)
        trainX, testX = scale_data(trainX, testX)
        outp, score, hist = evaluate(trainX, trainy, testX, testy, model, ai_settings)
        scores.append(score)
        print('>#%d: %.3f' % (r+1, score))
        hists.append(hist)   
        outps.append(outp)
    summarize_results(scores)
    return outps, scores, hists

def run_plot_standard(data, ntrials, noutputs, model):
    ''' Runs a standard algorithm, prints and plots the results. '''
    trainX, trainy, testX, testy, _, test_sample = split_data(data, ntrials, noutputs)
    model.fit(trainX, trainy)
    predy = model.predict(testX)
    
    # print results
    print("Mean absolute error: %.2f" % mean_absolute_error(testy, predy))
    print('Variance score: %.2f' % r2_score(testy, predy))
    print()
    predy02 = [ 0 if x<0.2 else x for x in predy]
    print('The mean absolute error is: ', mean_absolute_error(testy, predy))
    print('The mean absolute error when making everything below 0.2 zero: ', mean_absolute_error(testy, predy02))
    print('The mean absolute error when predicting 0 everywhere: ', mean_absolute_error([0]*len(testy),testy))
    print('The mean absolute error when predicting the mean everywhere: ', mean_absolute_error([trainy.mean()]*len(testy),testy))
    print('The mean absolute error when predicting the median everywhere: ', mean_absolute_error([np.median(trainy)]*len(testy),testy))
    r=[]
    for i in test_sample:
        r = r + [str(int(i/2))+' Right'] + [str(int(i/2))+' Left']
    # plot results
    plot_output(testy, predy, 'Model output', 'y', 'predict', r)

def run_plot_neural(data, ntrials, noutputs, model, ai_settings):
    ''' Runs the neural network, plots the training history and plots the results. '''
    trainX, trainy, testX, testy, _, _ = split_data(data, ntrials, noutputs)
    trainX, testX = scale_data(trainX, testX)
    outp, mae, hist = evaluate(trainX, trainy, testX, testy, model, ai_settings)
    print("Mean absolute error: %.4f" %mae)
    
    # plot training
    plt.figure(figsize=(20,10))
    plt.plot(hist.history['mean_absolute_error'])
    plt.plot(hist.history['val_mean_absolute_error'])
    plt.title('Model MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    
    # plot results
    if testy.ndim == 1:
        plot_output(testy, outp, 'Model output', 'y', 'predict')
    if testy.ndim == 2:
        ca_lower_test = list(map(list, zip(*testy)))[0]
        dys_lower_test = list(map(list, zip(*testy)))[1]
        ca_lower_pred = list(map(list, zip(*outp)))[0]
        dys_lower_pred = list(map(list, zip(*outp)))[1]
        plot_output(ca_lower_test, ca_lower_pred, 'Model CA output', 'CA y', 'CA predict')      
        plot_output(dys_lower_test, dys_lower_pred, 'Model DYS output', 'DYS y', 'DYS predict')

def evaluate(trainX, trainy, testX, testy, model, ai_settings):
    ''' Create, compile and evaluate the model. '''
    n_timesteps, n_features = trainX.shape[1], trainX.shape[2]
    model, n_heads = model(n_timesteps, n_features)
    (trainX, testX) = ([trainX]*n_heads, [testX]*n_heads) if n_heads > 1 else (trainX, testX)
    model.compile(loss='mean_squared_error', optimizer=ai_settings['optimizer'], metrics=['mean_absolute_error'])
    history = model.fit(trainX, trainy, epochs=ai_settings['epochs'], validation_split=ai_settings['validation_split'], batch_size=ai_settings['batch_size'], verbose=ai_settings['verbose'])
    _, accuracy = model.evaluate(testX, testy, batch_size=ai_settings['batch_size'], verbose=ai_settings['verbose'])
    outp = model.predict(testX)
    return outp, accuracy, history