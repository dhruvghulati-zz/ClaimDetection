'''

This file takes in the probability predictions for the listed models from the page before which were saved to a text file, if they have one, and outputs various possibilities of predictions based on various thresholds for the probabilities.

python src/main/probPredictor.py data/output/zero/test/ 0.05 data/output/zero/test/models.json data/output/zero/test/summaryEvaluation.csv

'''
from collections import Counter

import numpy as np
import pandas as pd
import sys
from sklearn.metrics import precision_score, precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import os
import json
import copy


def probabilityThreshold(categories,prediction, numberProperties, testCatLabels,fixedProbThreshold):
    print "Number of properties is", numberProperties
    meanProbThreshold = 1 / float(numberProperties)
    print "Threshold is", meanProbThreshold
    print "Prob threshold is", fixedProbThreshold

    # print "Categories are",categories
    # # print "prediction is", prediction
    # print "numberProperties is",numberProperties

    fixedPrediction = np.array(prediction)
    catPrediction = np.copy(fixedPrediction).tolist()

    print "Dimensions of prediction matrix are",fixedPrediction.shape

    catIndex = []

    probPrediction = np.copy(fixedPrediction)
    catPropPrediction = np.copy(probPrediction).tolist()

    # print type(catPropPrediction)
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

    # Binarise the probabilities in the fixed and other versions
    for i, sentence in enumerate(fixedPrediction):
        for j, prob in enumerate(sentence):
            if prob > meanProbThreshold:
                catPrediction[i][j] = categories[catIndex[j]]
                fixedPrediction[i][j] = 1
            else:
                catPrediction[i][j] = "no_region"
                fixedPrediction[i][j] = 0

    for i, sentence in enumerate(probPrediction):
        for j, prob in enumerate(sentence):
            if prob > fixedProbThreshold:
                catPropPrediction[i][j] = categories[catIndex[j]]
                probPrediction[i][j] = 1
            else:
                catPropPrediction[i][j] = "no_region"
                probPrediction[i][j] = 0

    # print "Binary labels are", prediction
    # print "Fixed binary labels are", probPrediction

    # print catPrediction

    # print "Number of 0s in fixed binary labels ", fixedPrediction.size - np.count_nonzero(fixedPrediction)
    # print "Number of 0s in variable binary labels are ", probPrediction.size - np.count_nonzero(probPrediction),"\n"

    # c1 = Counter([i for j in catPrediction for i in j])
    # c2 = Counter([i for j in catPropPrediction for i in j])
    #
    # # sum([len(arr) for arr in catPrediction]) -
    # # sum([len(arr) for arr in catPropPrediction]) -
    #
    # print "Number of no_regions in fixed category labels are ", c1['no_region']
    # print "Number of no_regions in variable category labels are ", c2['no_region']

    # print "Arranged catindex is ",np.arange(len(catIndex))

    predictedBinaryValues = []
    predictedBinaryValuesVar = []

    predictedCatValues = []
    predictedCatValuesVar = []

    for i,j in zip(np.arange(len(catIndex)),catIndex):
        if j==-1:
            predictedBinaryValues.append(0)
            predictedBinaryValuesVar.append(0)
            predictedCatValues.append("no_region")
            predictedCatValuesVar.append("no_region")
        else:
            predictedBinaryValues.append(fixedPrediction[i,j])
            predictedBinaryValuesVar.append(probPrediction[i,j])
            predictedCatValues.append(catPrediction[i][j])
            predictedCatValuesVar.append(catPropPrediction[i][j])

    return predictedBinaryValues, predictedBinaryValuesVar, predictedCatValues,predictedCatValuesVar

if __name__ == "__main__":

    print "loading from file " + sys.argv[3]
    # model_data = np.loadtxt(sys.argv[3])
    with open(sys.argv[3]) as modelFile:
        model_data = json.loads(modelFile.read())

    test = pd.read_csv(os.path.join(sys.argv[1] + 'testData.csv'))

    probThreshold = float(sys.argv[2])

    testSet = test['testSet'][0]

    y_pospred = np.ones(len(test))

    y_multi_true  = np.array(test['property'])
    y_true_claim = np.array(test['claim'])

    '''
    Now we load in the probability predictions for each model we care about
    '''

    prob_prediction_classes = np.loadtxt(os.path.join(sys.argv[1] +'open_categories.txt'),dtype=str)

    closed_prob_prediction_classes = np.loadtxt(os.path.join(sys.argv[1] +'closed_categories.txt'),dtype=str)



    for model, dict in model_data.iteritems():
        # print model
        if dict['prob_prediction']:
            dict['training_classes'] = np.shape(dict['prob_prediction'])[1]
            if model.startswith("open"):

                dict['prob_prediction_binary_fixed_threshold'] = probabilityThreshold(prob_prediction_classes,dict['prob_prediction'],dict['training_classes'], y_multi_true,probThreshold)[0]

                dict['prob_prediction_binary_variable_threshold'] = probabilityThreshold(prob_prediction_classes,dict['prob_prediction'],dict['training_classes'], y_multi_true,probThreshold)[1]

                dict['prob_prediction_categorical_fixed_threshold'] = probabilityThreshold(prob_prediction_classes,dict['prob_prediction'],dict['training_classes'], y_multi_true,probThreshold)[2]

                dict['prob_prediction_categorical_variable_threshold'] = probabilityThreshold(prob_prediction_classes,dict['prob_prediction'],dict['training_classes'], y_multi_true,probThreshold)[3]
            else:
                dict['prob_prediction_binary_fixed_threshold'] = probabilityThreshold(closed_prob_prediction_classes,dict['prob_prediction'],dict['training_classes'], y_multi_true,probThreshold)[0]

                dict['prob_prediction_binary_variable_threshold'] = probabilityThreshold(closed_prob_prediction_classes,dict['prob_prediction'],dict['training_classes'], y_multi_true,probThreshold)[1]

                dict['prob_prediction_categorical_fixed_threshold'] = probabilityThreshold(closed_prob_prediction_classes,dict['prob_prediction'],dict['training_classes'], y_multi_true,probThreshold)[2]

                dict['prob_prediction_categorical_variable_threshold'] = probabilityThreshold(closed_prob_prediction_classes,dict['prob_prediction'],dict['training_classes'], y_multi_true,probThreshold)[3]

    categorical_data = {}

    for model, dict in model_data.iteritems():
        if any(dict['prob_prediction']):
            categorical_data[(model, 'fixed_precision')] = \
            precision_recall_fscore_support(y_multi_true, dict['prob_prediction_categorical_fixed_threshold'], labels=list(set(y_multi_true)),
                                            average=None)[0]
            categorical_data[(model, 'fixed_recall')] = \
            precision_recall_fscore_support(y_multi_true, dict['prob_prediction_categorical_fixed_threshold'], labels=list(set(y_multi_true)),
                                            average=None)[1]
            categorical_data[(model, 'fixed_f1')] = \
            precision_recall_fscore_support(y_multi_true, dict['prob_prediction_categorical_fixed_threshold'], labels=list(set(y_multi_true)),
                                            average=None)[2]

            categorical_data[(model, 'variable_precision')] = \
            precision_recall_fscore_support(y_multi_true, dict['prob_prediction_categorical_variable_threshold'], labels=list(set(y_multi_true)),
                                            average=None)[0]
            categorical_data[(model, 'variable_recall')] = \
            precision_recall_fscore_support(y_multi_true, dict['prob_prediction_categorical_variable_threshold'], labels=list(set(y_multi_true)),
                                            average=None)[1]
            categorical_data[(model, 'variable_f1')] = \
            precision_recall_fscore_support(y_multi_true, dict['prob_prediction_categorical_variable_threshold'], labels=list(set(y_multi_true)),
                                            average=None)[2]


    categorical_data = pd.DataFrame(categorical_data, index=[item.split('/')[3] for item in list(set(y_multi_true))])

    categoricalPath = os.path.join(sys.argv[1] + testSet + "_" + 'no_threshold' + str(probThreshold)+'_categoricalResults.csv')

    categorical_data.to_csv(path_or_buf=categoricalPath, encoding='utf-8')

    output = {(outerKey, innerKey): values for outerKey, innerDict in model_data.iteritems() for innerKey, values in innerDict.iteritems() if innerKey=='prob_prediction_binary_fixed_threshold' or innerKey.startswith('prob_prediction_binary_') or innerKey.startswith('prob_prediction_categorical_')}

    output = pd.DataFrame(output)

    # print "Output is ",output

    parsed_sentences =  np.array(test['parsedSentence'])
    threshold_array= np.array(["no_threshold"] * len(parsed_sentences))
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

    resultPath = os.path.join(sys.argv[1]+ testSet + '_' + str("no_threshold") + '_' + str(probThreshold)+ '_regressionResult.csv')

    output.to_csv(path_or_buf=resultPath,encoding='utf-8')


    summaryDF = pd.DataFrame()

    def evaluation(trueLabels, evalLabels, test_set,probThreshold):
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
                'ape_threshold': ["no_mape_threshold"],
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
        evaluation(y_true_claim, result, testSet, probThreshold)

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