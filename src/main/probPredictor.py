'''

This file takes in the probability predictions for the listed models from the page before which were saved to a text file, if they have one, and outputs various possibilities of predictions based on various thresholds for the probabilities.

python src/main/probPredictor.py data/output/zero/test/ 0.05 data/output/zero/test/models.json data/output/zero/test/summaryEvaluation.csv

'''

import numpy as np
import pandas as pd
import sys
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import os
import json


def probabilityThreshold(categories,prediction, numberProperties, testCatLabels,fixedProbThreshold):
    print "Number of properties is", numberProperties
    meanProbThreshold = 1 / float(numberProperties)
    print "Threshold is", meanProbThreshold
    print "Prob threshold is", fixedProbThreshold

    prediction = np.array(prediction)

    print "Dimensions of prediction matrix are",prediction.shape

    catIndex = []

    probPrediction = np.copy(prediction)

    # print type(probPrediction)
    # print "Probabilities are", probPrediction

    for i, label in enumerate(testCatLabels):
        # If this doesn't find an index for the test labels, it means we haven't found a binary label for that property, wasn't even in our prediction, so actually should be given the index for no_region'
        index = [i for i, s in enumerate(categories) if s == label]
        if not index:
            index = [-1]
        catIndex.append(index)

    # print "Categorical indices pre-ravel are", catIndex

    catIndex = np.array(catIndex).ravel()
    # print "New categorical indices are", catIndex
    # print type(catIndex)

    # Binarise the probabilities
    for i, sentence in enumerate(prediction):
        for j, prob in enumerate(sentence):
            if prob > meanProbThreshold:
                prediction[i][j] = 1
            else:
                prediction[i][j] = 0

    for i, sentence in enumerate(probPrediction):
        for j, prob in enumerate(sentence):
            if prob > fixedProbThreshold:
                probPrediction[i][j] = 1
            else:
                probPrediction[i][j] = 0

    # print "Binary labels are", prediction
    # print "Fixed binary labels are", probPrediction

    print "Number of 0s are ", prediction.size - np.count_nonzero(prediction)
    print "Number of 0s in fixed binary labels are ", probPrediction.size - np.count_nonzero(probPrediction)

    # print "Arranged catindex is ",np.arange(len(catIndex))

    predictedBinaryValues = []
    argProbResultBinary = []

    for i,j in zip(np.arange(len(catIndex)),catIndex):
        if j==-1:
            predictedBinaryValues.append(0)
            argProbResultBinary.append(0)
        else:
            predictedBinaryValues.append(prediction[i,j])
            argProbResultBinary.append(probPrediction[i,j])

    return predictedBinaryValues, argProbResultBinary

if __name__ == "__main__":

    print "loading from file " + sys.argv[3]
    # model_data = np.loadtxt(sys.argv[3])
    with open(sys.argv[3]) as modelFile:
        model_data = json.loads(modelFile.read())

    test = pd.read_csv(os.path.join(sys.argv[1] + '/testData.csv'))

    probThreshold = float(sys.argv[2])

    testSet = test['testSet'][0]

    threshold = test['threshold'][0]

    y_pospred = np.ones(len(test))

    y_multi_true  = np.array(test['property'])
    y_true_claim = np.array(test['claim'])

    '''
    Now we load in the probability predictions for each model we care about
    '''

    # prob_prediction = np.loadtxt(os.path.join(sys.argv[1] +'/open_prob_a.txt'))
    #
    # prob_prediction_threshold = np.loadtxt(os.path.join(sys.argv[1] +'/open_prob_a_threshold.txt'))
    #
    # closed_prob_prediction = np.loadtxt(os.path.join(sys.argv[1] +'/closed_prob_a.txt'))
    #
    # closed_prob_prediction_threshold = np.loadtxt(os.path.join(sys.argv[1] +'/closed_prob_a_threshold.txt'))

    prob_prediction_classes = np.loadtxt(os.path.join(sys.argv[1] +'/open_categories.txt'),dtype=str)

    prob_prediction_threshold_classes = np.loadtxt(os.path.join(sys.argv[1] +'/openthreshold_categories.txt'),dtype=str)

    closed_prob_prediction_classes = np.loadtxt(os.path.join(sys.argv[1] +'/closed_categories.txt'),dtype=str)

    closed_prob_prediction_threshold_classes = np.loadtxt(os.path.join(sys.argv[1] +'/closedthreshold_categories.txt'),dtype=str)

    for model, dict in model_data.iteritems():
        # print model
        if dict['prob_prediction']:
            dict['training_classes'] = np.shape(dict['prob_prediction'])[1]
            if model.startswith("open"):
                if str(model).split('_')[-2:-1]=="threshold":
                    dict['prob_prediction_binary_variable_threshold'] = probabilityThreshold(prob_prediction_threshold_classes,dict['prob_prediction'],dict['training_classes'], y_multi_true,probThreshold)[0]
                    dict['prob_prediction_binary_fixed_threshold'] = probabilityThreshold(prob_prediction_threshold_classes,dict['prob_prediction'],dict['training_classes'], y_multi_true,probThreshold)[1]
                else:
                    dict['prob_prediction_binary_variable_threshold'] = probabilityThreshold(prob_prediction_classes,dict['prob_prediction'],dict['training_classes'], y_multi_true,probThreshold)[0]
                    dict['prob_prediction_binary_fixed_threshold'] = probabilityThreshold(prob_prediction_classes,dict['prob_prediction'],dict['training_classes'], y_multi_true,probThreshold)[1]
            else:
                if str(model).split('_')[-2:-1]=="threshold":
                    dict['prob_prediction_binary_variable_threshold'] = probabilityThreshold(closed_prob_prediction_threshold_classes,dict['prob_prediction'],dict['training_classes'], y_multi_true,probThreshold)[0]
                    dict['prob_prediction_binary_fixed_threshold'] = probabilityThreshold(closed_prob_prediction_threshold_classes,dict['prob_prediction'],dict['training_classes'], y_multi_true,probThreshold)[1]
                else:
                    dict['prob_prediction_binary_variable_threshold'] = probabilityThreshold(closed_prob_prediction_classes,dict['prob_prediction'],dict['training_classes'], y_multi_true,probThreshold)[0]
                    dict['prob_prediction_binary_fixed_threshold'] = probabilityThreshold(closed_prob_prediction_classes,dict['prob_prediction'],dict['training_classes'], y_multi_true,probThreshold)[1]


    output = {(outerKey, innerKey): values for outerKey, innerDict in model_data.iteritems() for innerKey, values in innerDict.iteritems() if innerKey=='prob_prediction_binary_fixed_threshold' or innerKey.startswith('prob_prediction_binary_')}

    output = pd.DataFrame(output)

    # print "Output is ",output

    parsed_sentences =  np.array(test['parsedSentence'])
    threshold_array= np.array([threshold] * len(parsed_sentences))
    claim_array= np.array(test['claim'])

    data = {
            ('global','parsed_sentence'): parsed_sentences,
            ('global','threshold'):threshold_array,
            ('global','claim'):claim_array
            }

    # print "Data is",data

    DF = pd.DataFrame(data)

    # print "Data frame is",DF

    output = pd.concat([output,DF],axis=1)

    resultPath = os.path.join(sys.argv[1]+ testSet + '_' + str(threshold) + '_' + str(probThreshold)+ '_regressionResult.csv')

    output.to_csv(path_or_buf=resultPath,encoding='utf-8')


    summaryDF = pd.DataFrame(columns=('precision', 'recall', 'f1', 'accuracy', 'evaluation set', 'threshold','probThreshold'))

    def evaluation(trueLabels, evalLabels, test_set, threshold,probThreshold):
        # ,probThreshold
        global summaryDF
        global trainingLabels
        global positiveOpenTrainingLabels
        global negativeOpenTrainingLabels
        global positiveClosedTrainingLabels
        global negativeClosedTrainingLabels
        global openTrainingClasses
        global openTrainingClassesThreshold
        global closedTrainingClasses
        global closedTrainingClassesThreshold

        precision = precision_score(trueLabels, evalLabels)
        recall = recall_score(trueLabels, evalLabels)
        f1 = f1_score(trueLabels, evalLabels)
        accuracy = accuracy_score(trueLabels, evalLabels)

        data = {'precision': [precision],
                'recall': [recall],
                'f1': [f1],
                'accuracy': [accuracy],
                'evaluation set': [test_set],
                'threshold': [threshold],
                'probThreshold': [probThreshold],
                'trainingLabels':[""],
                'positiveOpenLabels':[""],
                'negativeOpenLabels':[""],
                'positiveClosedLabels':[""],
                'negativeClosedLabels':[""],
                'openTrainingClasses': [""],
                'openTrainingClassesThreshold': [""],
                'closedTrainingClasses':[""],
                'closedTrainingClassesThreshold': [""]
                }

        DF = pd.DataFrame(data)

        summaryDF = pd.concat([summaryDF, DF])


    results = []
    columns = []

    for model,dict in model_data.iteritems():
        for key in dict.keys():
            if key =='prob_prediction_binary_fixed_threshold':
                # print "Model is",model
                results.append(dict['prob_prediction_binary_fixed_threshold'])
                columns.append(model+"_fixed")
            if key =='prob_prediction_binary_variable_threshold':
                # print "Model is",model
                results.append(dict['prob_prediction_binary_variable_threshold'])
                columns.append(model+"_"+str(probThreshold)+"_threshold")


    for result in results:
        evaluation(y_true_claim, result, testSet, threshold, probThreshold)

    # print summaryDF

    columns = list(columns)

    summaryDF.index = columns
    #
    # print summaryDF
    #
    try:
        if os.stat(sys.argv[4]).st_size > 0:
            with open(sys.argv[4], 'a') as f:
                # Need to empty file contents now
                f.write('\n')
                summaryDF.to_csv(path_or_buf=f, encoding='utf-8', mode='a', header=False)
        else:
            print "empty file"
            with open(sys.argv[4], 'w+') as f:
                summaryDF.to_csv(path_or_buf=f, encoding='utf-8')
    except OSError:
        print "No file"
        with open(sys.argv[4], 'w+') as f:
            summaryDF.to_csv(path_or_buf=f, encoding='utf-8')