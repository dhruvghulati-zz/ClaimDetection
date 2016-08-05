import numpy as np
import pandas as pd
import sys
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import os

rng = np.random.RandomState(101)


def probabilityThreshold(categories,prediction, numberProperties, testCatLabels,fixedProbThreshold):
    print "Number of properties is", numberProperties
    meanProbThreshold = 1 / float(numberProperties)
    print "Threshold is", meanProbThreshold
    print "Prob threshold is", fixedProbThreshold

    # print testCatLabels

    print "Categories are", categories

    # print "Probabilities are", prediction

    print "Dimensions of prediction matrix are",prediction.shape

    catIndex = []

    probPrediction = np.copy(prediction)

    print type(probPrediction)
    # print "Probabilities are", probPrediction

    for i, label in enumerate(testCatLabels):
        # If this doesn't find an index for the test labels, it means we haven't found a binary label for that property, wasn't even in our prediction, so actually should be given the index for no_region'
        index = [i for i, s in enumerate(categories) if s == label]
        if not index:
            index = [-1]
        catIndex.append(index)

    # print "Categorical indices pre-ravel are", catIndex

    catIndex = np.array(catIndex).ravel()
    print "New categorical indices are", catIndex
    print type(catIndex)

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

    # TODO - the predicted binary labels in prediction don't match the categorical index of the categories being predicted for. It should match the unique labels?

    predictedBinaryValues = []
    argProbResultBinary = []

    for i,j in zip(np.arange(len(catIndex)),catIndex):
        if j==-1:
            predictedBinaryValues.append(0)
            argProbResultBinary.append(0)
        else:
            predictedBinaryValues.append(prediction[i,j])
            argProbResultBinary.append(probPrediction[i,j])

    print "Predicted binary values are", predictedBinaryValues
    print "Predicted fixed binary values are", argProbResultBinary
    # prediction[np.arange(len(catIndex)), catIndex if not catIndex=-1 else 0]
    # predictedCats = categories[catIndex]
    # argProbResultBinary = probPrediction[np.arange(len(catIndex)), catIndex]
    #
    # catResult = np.where(predictedBinaryValues, predictedCats, "no_region")
    # argProbResult = np.where(argProbResultBinary, predictedCats, "no_region")

    # print catResult

    return predictedBinaryValues, argProbResultBinary

if __name__ == "__main__":

    probThreshold = float(sys.argv[3])

    test = pd.read_csv(os.path.join(sys.argv[1] + '/testData.csv'))

    testSet = test['testSet'][0]

    threshold = test['threshold'][0]

    y_pospred = np.ones(len(test))

    y_multi_true  = np.array(test['property'])
    y_true_claim = np.array(test['claim'])

    prob_prediction = np.loadtxt(os.path.join(sys.argv[1] +'/open_prob_a.txt'))

    prob_prediction_threshold = np.loadtxt(os.path.join(sys.argv[1] +'/open_prob_a_threshold.txt'))

    closed_prob_prediction = np.loadtxt(os.path.join(sys.argv[1] +'/closed_prob_a.txt'))

    closed_prob_prediction_threshold = np.loadtxt(os.path.join(sys.argv[1] +'/closed_prob_a_threshold.txt'))

    prob_prediction_classes = np.loadtxt(os.path.join(sys.argv[1] +'/open_categories.txt'),dtype=str)

    prob_prediction_threshold_classes = np.loadtxt(os.path.join(sys.argv[1] +'/openthreshold_categories.txt'),dtype=str)

    closed_prob_prediction_classes = np.loadtxt(os.path.join(sys.argv[1] +'/closed_categories.txt'),dtype=str)

    closed_prob_prediction_threshold_classes = np.loadtxt(os.path.join(sys.argv[1] +'/closedthreshold_categories.txt'),dtype=str)


    # TODO = get the length of each prediction as its number of labels

    openTrainingClasses = np.shape(prob_prediction)[1]
    openTrainingClassesThreshold = np.shape(prob_prediction_threshold)[1]
    closedTrainingClasses = np.shape(closed_prob_prediction)[1]
    closedTrainingClassesThreshold = np.shape(closed_prob_prediction_threshold)[1]


    print "Predicting open multinomial test labels with/without MAPE threshold using probability based predictor...\n"

    y_multi_logit_result_open_prob_binary,y_multi_logit_result_open_prob_binaryFixed = probabilityThreshold(prob_prediction_classes,prob_prediction,np.shape(prob_prediction)[1], y_multi_true,probThreshold)

    y_multi_logit_result_open_prob_binary_threshold,y_multi_logit_result_open_prob_binary_thresholdFixed = probabilityThreshold(prob_prediction_threshold_classes,prob_prediction_threshold,np.shape(prob_prediction_threshold)[1], y_multi_true,probThreshold)

    print "Predicting closed multinomial test labels with/without MAPE threshold using probability based predictor...\n"
    y_multi_logit_result_closed_prob_binary, y_multi_logit_result_closed_prob_binaryFixed = probabilityThreshold(closed_prob_prediction_classes,closed_prob_prediction,np.shape(closed_prob_prediction)[1],y_multi_true,probThreshold)

    y_multi_logit_result_closed_prob_binary_threshold,y_multi_logit_result_closed_prob_binary_thresholdFixed = probabilityThreshold(closed_prob_prediction_threshold_classes,closed_prob_prediction_threshold,np.shape(closed_prob_prediction_threshold)[1],y_multi_true,probThreshold)

    output = pd.DataFrame(data=dict(parsed_sentence=test['parsedSentence'],

                                    open_property_probability_prediction_toBinary=y_multi_logit_result_open_prob_binary,
                                    open_property_probability_threshold_prediction_toBinary=y_multi_logit_result_open_prob_binary_threshold,
                                    closed_property_probability_prediction_toBinary=y_multi_logit_result_closed_prob_binary,
                                    closed_property_probability_threshold_prediction_toBinary=y_multi_logit_result_closed_prob_binary_threshold,

                                    open_property_probability_prediction_toBinaryFixed=y_multi_logit_result_open_prob_binaryFixed,
                                    open_property_probability_threshold_prediction_toBinaryFixed=y_multi_logit_result_open_prob_binary_thresholdFixed,

                                    closed_property_probability_prediction_toBinaryFixed=y_multi_logit_result_closed_prob_binaryFixed,
                                    closed_property_probability_threshold_prediction_toBinaryFixed=y_multi_logit_result_closed_prob_binary_thresholdFixed,

                                    test_data_mape_label=test['mape_label'],
                                    claim_label=y_true_claim,
                                    test_data_property_label=test['property'],
                                    andreas_prediction=y_pospred,
                                    threshold=np.full(len(y_true_claim), threshold),
                                    probThreshold = np.full(len(y_true_claim), probThreshold)
                                    ))



    resultPath = os.path.join(sys.argv[1] +'/'+testSet + '_' + str(threshold) + '_' + str(probThreshold)+ '_regressionResult.csv')

    output.to_csv(path_or_buf=resultPath, encoding='utf-8', index=False, cols=[
        'parsed_sentence',
        'features',

        'open_property_prediction_withMAPEthreshold',
        'open_property_prediction_withMAPEthreshold_toBinary',
        'closed_property_prediction_withMAPEthreshold',
        'closed_property_prediction_withMAPEthreshold_toBinary',

        'open_property_probability_prediction_toBinary',
        'open_property_probability_threshold_prediction_toBinary',

        'closed_property_probability_prediction_toBinary',
        'closed_property_probability_threshold_prediction_toBinary',

        'open_property_probability_prediction_toBinaryFixed',
        'open_property_probability_threshold_prediction_toBinaryFixed',

        'closed_property_probability_prediction_toBinaryFixed',
        'closed_property_probability_threshold_prediction_toBinaryFixed',


        'test_data_mape_label',
        'claim_label',
        'andreas_property_label',
        'andreas_prediction',
        'threshold',
        'probThreshold'
    ])

    # TODO - need to create a per property chart

    # Now we write our precision F1 etc to an Excel file
    summaryDF = pd.DataFrame(columns=('precision', 'recall', 'f1', 'accuracy', 'evaluation set', 'threshold','probThreshold'))


    def evaluation(trueLabels, evalLabels, test_set, threshold,probThreshold):
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
                'openTrainingClasses': [openTrainingClasses],
                'openTrainingClassesThreshold': [openTrainingClassesThreshold],
                'closedTrainingClasses':[closedTrainingClasses],
                'closedTrainingClassesThreshold': [closedTrainingClassesThreshold],
                }

        DF = pd.DataFrame(data)

        summaryDF = pd.concat([summaryDF, DF])


    results = [
               y_multi_logit_result_open_prob_binary,
               y_multi_logit_result_open_prob_binary_threshold,
               y_multi_logit_result_closed_prob_binary,
               y_multi_logit_result_closed_prob_binary_threshold,

               y_multi_logit_result_open_prob_binaryFixed,
               y_multi_logit_result_open_prob_binary_thresholdFixed,
               y_multi_logit_result_closed_prob_binaryFixed,
               y_multi_logit_result_closed_prob_binary_thresholdFixed

               ]
    #

    for result in results:
        evaluation(y_true_claim, result, testSet, threshold,probThreshold)

    columns = list([
                    'Open_Property_Probability_Prediction',
                    'Open_Property_Probability_Prediction_MAPEThreshold',
                    'Closed_Property_Probability_Prediction',
                    'Closed_Property_Probability_Prediction_MAPEThreshold',
                    'Open_Property_Probability_PredictionFixed',
                    'Open_Property_Probability_Prediction_MAPEThresholdFixed',
                    'Closed_Property_Probability_PredictionFixed',
                    'Closed_Property_Probability_Prediction_MAPEThresholdFixed'
                    ])

    # summaryDF.set_index(['A','B'])
    summaryDF.index = columns

    print summaryDF
    #


    try:
        if os.stat(sys.argv[2]).st_size > 0:
            # df_csv = pd.read_csv(sys.argv[5],encoding='utf-8',engine='python')
            # summaryDF = pd.concat([df_csv,summaryDF],axis=1,ignore_index=True)
            with open(sys.argv[2], 'a') as f:
                # Need to empty file contents now
                summaryDF.to_csv(path_or_buf=f, encoding='utf-8', mode='a', header=False)
                f.close()
        else:
            print "empty file"
            with open(sys.argv[2], 'w+') as f:
                summaryDF.to_csv(path_or_buf=f, encoding='utf-8')
                f.close()
    except OSError:
        print "No file"
        with open(sys.argv[2], 'w+') as f:
            summaryDF.to_csv(path_or_buf=f, encoding='utf-8')
            f.close()