'''
This calculates features in the same way as a multinomial predictor, but generates the input data files in the format needed for a classifier.

The input parameters come from:

- The APE threshold applied to the predictor and test features
- A cost threshold (can varied)
- Bias term of a sigmoid function
- Slope term of sigmoid.

Note, this only looks at wordgrams but this can be extended.

python src/main/costSensitiveClassifier.py data/output/predictedPropertiesZero.json data/output/devLabels.json data/featuresKept.json data/output/zero/arow_test/ [cost] [bias] [slope] ![data/output/zero/arow_test/costMatrices.json]

'''

import math
import arow
import re
from nltk.corpus import stopwords  # Import the stop word list
import json
import numpy as np
import copy
import sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import os
from sklearn.pipeline import Pipeline
from collections import defaultdict
xs = defaultdict(list)
from numpy import inf
from scipy.special import expit
from nltk.tokenize import WhitespaceTokenizer

# print sys.getdefaultencoding()

def sigmoid(b,m,v):
    return expit(b + m*v)*2 - 1

def sentence_to_words(sentence, remove_stopwords=False):
    letters_only = re.sub('[^a-zA-Z| LOCATION_SLOT | NUMBER_SLOT]', " ", sentence)
    words = letters_only.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return (words)


def genCostMatrices(inputSentences, costMatrix, openVector, closedVector, openThresholdVector, closedThresholdVector,
                    inputCostThreshold, inputBias, inputSlope):

    for i, sentence in enumerate(inputSentences):
        if sentence:

            # Generate all the variables I need
            error = float(sentence['meanAbsError'])
            closedError = float(sentence['closedMeanAbsError'])

            openDict = sentence['openCostDict']
            closedDict = sentence['closedCostDict']

            for key,value in openDict.items():
                if value == inf:
                    openDict[key] = 1e10

            for key,value in closedDict.items():
                if value == inf:
                    openDict[key] = 1e10

            # Now we ensure there is a value for each potential label in the cost vector
            for key, value in openVector.items():
                if key not in openDict:
                    openDict[key] = 1e10

            for key, value in closedVector.items():
                if key not in closedDict:
                    closedDict[key] = 1e10

            '''
            Now we add in the APE threshold which conditionally adds a void label
            '''

            openDictThreshold = copy.deepcopy(openDict)
            closedDictThreshold = copy.deepcopy(closedDict)

            # Now we add a threshold cost vector which extends the normal one to add a new label, no_region which is 0 or 1e10inity depending on the results of the MAPE threshold
            openDictThreshold['no_region'] = float(0) if sentence[
                                                             'predictedPropertyOpenThreshold'] == "no_region" else 1e10
            closedDictThreshold['no_region'] = float(0) if sentence[
                                                               'predictedPropertyClosedThreshold'] == "no_region" else 1e10

            extracted = sentence['location-value-pair'].values()[0]
            # No need to account for zero values as not dividing
            if extracted == 0:
                extracted == 0e-10

            # print "Extracted is",extracted
            '''
            Get the chosen predictions and their minimum values - this is for the single array case.
            '''

            openPrediction = sentence['predictedPropertyOpen']
            closedPrediction = sentence['predictedPropertyClosed']

            '''
            Calculate the best choice
            '''

            openPredictionThreshold = min(openDictThreshold, key=openDictThreshold.get)
            closedPredictionThreshold = min(closedDictThreshold, key=closedDictThreshold.get)

            '''
            Now we create a layer on the top - cost thresholded versions of the above
            '''

            openDictCostThreshold = {key: (1e10 if value > inputCostThreshold else value) for key, value in
                                     openDict.iteritems()}
            closedDictCostThreshold = {key: (1e10 if value > inputCostThreshold else value) for key, value in
                                       closedDict.iteritems()}

            openDictThresholdCostThreshold = {key: (1e10 if value > inputCostThreshold else value) for key, value in
                                              openDictThreshold.iteritems()}
            closedDictThresholdCostThreshold = {key: (1e10 if value > inputCostThreshold else value) for key, value in
                                                closedDictThreshold.iteritems()}

            '''
            Calculate the best choice
            '''

            openPredictionCostThreshold = min(openDictCostThreshold, key=openDictCostThreshold.get)
            closedPredictionCostThreshold = min(closedDictCostThreshold, key=closedDictCostThreshold.get)

            openPredictionThresholdCostThreshold = min(openDictThresholdCostThreshold,
                                                       key=openDictThresholdCostThreshold.get)
            closedPredictionThresholdCostThreshold = min(closedDictThresholdCostThreshold,
                                                         key=closedDictThresholdCostThreshold.get)




            '''
            Use calculations on the normalised format for the single version, and normalise also
            '''

            for key, data in costMatrix.iteritems():
                # When there is only one cost per training instance
                if str(key).startswith("open"):
                    if str(key).endswith("1"):
                        dict = {key: (0 if key == openPrediction else 1) for key, value in openVector.items()}
                        data['cost_matrix'].append(dict)
                    if str(key).endswith("2"):
                        dict = {key: sigmoid(inputBias,inputSlope,value) for key, value in openDict.items()}
                        data['cost_matrix'].append(dict)
                if str(key).startswith("closed"):
                    if str(key).endswith("1"):
                        dict = {key: (0 if key == closedPrediction else 1) for key, value in closedVector.items()}
                        data['cost_matrix'].append(dict)
                    if str(key).endswith("2"):
                        dict = {key: sigmoid(inputBias,inputSlope,value) for key, value in closedDict.items()}
                        data['cost_matrix'].append(dict)
                if str(key).startswith("threshold"):
                    if str(key).split('_')[1] == "open":
                        if str(key).endswith("1"):
                            dict = {key: (0 if key == openPredictionThreshold else 1) for key, value in openThresholdVector.items()}
                            data['cost_matrix'].append(dict)
                        if str(key).endswith("2"):
                            dict = {key: sigmoid(inputBias,inputSlope,value) for key, value in openDictThreshold.items()}
                            data['cost_matrix'].append(dict)
                    else:
                        if str(key).endswith("1"):
                            dict = {key: (0 if key == closedPredictionThreshold else 1) for key, value in closedThresholdVector.items()}
                            data['cost_matrix'].append(dict)
                        if str(key).endswith("2"):
                            dict = {key: sigmoid(inputBias,inputSlope,value) for key, value in closedDictThreshold.items()}
                            data['cost_matrix'].append(dict)
                if str(key).startswith("costThreshold"):
                    # Not a threshold case
                    if str(key).split('_')[1] == "open":
                        if str(key).endswith("1"):
                            dict = {key: (0 if key == openPredictionCostThreshold else 1) for key, value in openVector.items()}
                            data['cost_matrix'].append(dict)
                        if str(key).endswith("2"):
                            dict = {key: sigmoid(inputBias,inputSlope,value) for key, value in openDictCostThreshold.items()}
                            data['cost_matrix'].append(dict)
                    else:
                        if str(key).endswith("1"):
                            dict = {key: (0 if key == closedPredictionCostThreshold else 1) for key, value in closedVector.items()}
                            data['cost_matrix'].append(dict)
                        if str(key).endswith("2"):
                            dict = {key: sigmoid(inputBias,inputSlope,value) for key, value in closedDictCostThreshold.items()}
                            data['cost_matrix'].append(dict)
                if str(key).startswith("thresholdCostThreshold"):
                    if str(key).split('_')[1] == "open":
                        if str(key).endswith("1"):
                            dict = {key: (0 if key == openPredictionThresholdCostThreshold else 1) for key, value in openThresholdVector.items()}
                            data['cost_matrix'].append(dict)
                        if str(key).endswith("2"):
                            dict = {key: sigmoid(inputBias,inputSlope,value) for key, value in openDictThresholdCostThreshold.items()}
                            data['cost_matrix'].append(dict)
                    else:
                        if str(key).endswith("1"):
                            dict = {key: (0 if key == closedPredictionThresholdCostThreshold else 1) for key, value in closedThresholdVector.items()}
                            data['cost_matrix'].append(dict)
                        if str(key).endswith("2"):
                            dict = {key: sigmoid(inputBias,inputSlope,value) for key, value in closedDictThresholdCostThreshold.items()}
                            data['cost_matrix'].append(dict)

# for key, value in costMatrix.iteritems():
#     print key," array is ", value['cost_matrix'],"\n"

def generatePredictions(fullCostDict, trainWordgramFeatures, testWordGramFeatures):

    for model, dict in fullCostDict.items():
        print "Model being trained and predicted is:",model,"\n"

        if 'cost_matrix' in dict:

            f = open(dict['wordgrams_predict_path'], 'w')
            fp = open(dict['wordgrams_prob_path'], 'w')

            train_data = [arow.train_instance_from_list(costDict,wordgramsTrain) for (costDict, wordgramsTrain) in zip(dict['cost_matrix'],trainWordgramFeatures)]
            test_data = [arow.test_instance_from_list(wordgramsTest) for wordgramsTest in testWordGramFeatures]
            cl = arow.AROW()
            # print [cl.predict(d).label for d in test_data]
            # print [d.costs for d in test_data]

            cl.train(train_data)
            # cl.probGeneration()
            # ,probabilities=False
            predictions = [cl.predict(d, verbose=True).label for d in test_data]
            predictionScores = [cl.predict(d, verbose=True).label2score for d in test_data]

            for i, prediction in enumerate(predictions):
                line = str(int(prediction)) + " " + str(i)
                f.write(line+"\n")

            f.close()

            # Now normalise the scores
            # predictionScores = normalize_dicts_local(predictionScores)

            for i, prediction in enumerate(predictionScores):
                line = ""
                for label, score in prediction.iteritems():
                    line += str(int(label)) + ":" + str("{0:.3f}".format(score)) + " "
                fp.write(line + str(i)+"\n")

            fp.close()

def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])


def training_features(inputSentences):
    global vectorizer
    global train_wordbigram_list
    print "Preparing the raw features...\n"
    for i, sentence in enumerate(inputSentences):
        if sentence:
            words = (" ").join(sentence_to_words(sentence['parsedSentence'], True))
            word_list = sentence_to_words(sentence['parsedSentence'], True)
            bigrams = ""
            if "depPath" in sentence.keys():
                bigrams = [("+").join(bigram).encode('utf-8') for bigram in sentence['depPath']]
                bigrams = (' ').join(map(str, bigrams))
            bigrams = ('').join(bigrams)
            train_bigram_list.append(bigrams)
            # print "Sentence is ",type(words + " " + bigrams.decode("utf-8"))
            train_wordbigram_list.append(words + " " + bigrams.decode("utf-8"))
            train_wordlist.append(words)
            wordgrams = find_ngrams(word_list, 2)
            for i, grams in enumerate(wordgrams):
                wordgrams[i] = '+'.join(grams)
            wordgrams = (" ").join(wordgrams)
            train_gramlist.append(wordgrams)
            train_property_labels.append(sentence['predictedPropertyOpen'])
            train_property_labels_threshold.append(sentence['predictedPropertyOpenThreshold'])
            closed_train_property_labels.append(sentence['predictedPropertyClosed'])
            closed_train_property_labels_threshold.append(sentence['predictedPropertyClosedThreshold'])

    train_wordbigram_list = vectorizer.fit_transform(train_wordbigram_list).toarray().astype(np.float)


def test_features(testSentences):
    global vectorizer
    global test_wordbigram_list
    print "Preparing the raw test features...\n"
    for sentence in testSentences:
        if sentence['parsedSentence'] != {} and sentence['mape_label'] != {}:
            words = " ".join(sentence_to_words(sentence['parsedSentence'], True))
            word_list = sentence_to_words(sentence['parsedSentence'], True)
            wordgrams = find_ngrams(word_list, 2)
            for i, grams in enumerate(wordgrams):
                wordgrams[i] = '+'.join(grams)
            wordgrams = (" ").join(wordgrams)
            test_gramlist.append(wordgrams)
            test_wordlist.append(words)
            test_property_labels.append(sentence['property'])
            bigrams = sentence['depPath']
            test_bigram_list.append(bigrams)
            test_wordbigram_list.append((words + " " + bigrams.decode("utf-8")).decode("utf-8"))

    test_wordbigram_list = vectorizer.transform(test_wordbigram_list).toarray().astype(np.float)

if __name__ == "__main__":
    # training data
    # load the sentence file for training
    with open(sys.argv[1]) as trainingSentences:
        pattern2regions = json.loads(trainingSentences.read())

    print "We have ", len(pattern2regions), " training sentences."
    # We load in the allowable features and also no_region

    with open(sys.argv[2]) as testSentences:
        testSentences = json.loads(testSentences.read())

    print "Length of final sentences is", len(testSentences)

    threshold = testSentences[0]['threshold']

    print "APE Threshold for no region label is ", threshold

    with open(sys.argv[3]) as featuresKept:
        properties = json.loads(featuresKept.read())
    # properties.append("no_region")
    # print "Properties are ", properties

    # This is how we threshold across costs
    costThreshold = float(sys.argv[7])
    print "Cost Threshold is ", costThreshold

    # This is how we threshold across costs
    bias = float(sys.argv[8])
    print "Bias for sigmoid is ", bias

    # This is how we threshold across costs
    slope = float(sys.argv[9])
    print "Slope for sigmoid is ", slope


    '''
    Here are the global features
    '''
    train_wordlist = []
    test_wordlist = []

    train_gramlist = []
    test_gramlist = []

    train_bigram_list = []
    test_bigram_list = []

    train_wordbigram_list = []
    test_wordbigram_list = []

    '''
    Here are the global labels
    '''

    train_property_labels = []
    train_property_labels_threshold = []
    closed_train_property_labels = []
    closed_train_property_labels_threshold = []

    train_property_labels_depgrams = []
    train_property_labels_threshold_depgrams = []
    closed_train_property_labels_depgrams = []
    closed_train_property_labels_threshold_depgrams = []

    test_property_labels = []
    test_property_labels_depgrams = []

    # Vectorize in the same way as logistic

     # ('vect', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
    #                       ('tfidf', TfidfTransformer(use_idf=True,norm='l2',sublinear_tf=True)),
    vectorizer = Pipeline([('vect', CountVectorizer(analyzer="word",token_pattern="[\S]+",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),('tfidf', TfidfTransformer(use_idf=True,norm='l2',sublinear_tf=True))])

    # vectorizer = CountVectorizer(analyzer="word",token_pattern="[\S]+",stop_words=None,tokenizer=None,preprocessor=None,max_features=5000)

    print "Getting all the words in the training sentences...\n"

    # This both sorts out the features and the training labels
    training_features(pattern2regions)

    print len(train_wordlist), "sets of training features"

    trainingLabels = len(pattern2regions)
    positiveOpenTrainingLabels = len(train_property_labels_threshold) - train_property_labels_threshold.count(
        "no_region")
    negativeOpenTrainingLabels = train_property_labels_threshold.count("no_region")
    positiveClosedTrainingLabels = len(
        closed_train_property_labels_threshold) - closed_train_property_labels_threshold.count(
        "no_region")
    negativeClosedTrainingLabels = closed_train_property_labels_threshold.count(
        "no_region")

    # Fit the logistic classifiers to the training set, using the bag of words as features

    openTrainingClasses = len(set(train_property_labels))
    openTrainingClassesThreshold = len(set(train_property_labels_threshold))
    closedTrainingClasses = len(set(closed_train_property_labels))
    closedTrainingClassesThreshold = len(set(closed_train_property_labels_threshold))

    print "Get a bag of words for the test set, and convert to a numpy array\n"

    test_features(testSentences)

    print len(test_wordlist), "sets of test features"

    print "There are ", len(set(train_property_labels)), "open training classes"
    print "There are ", len(set(train_property_labels_threshold)), "open training classes w/threshold"
    print "There are ", len(set(closed_train_property_labels)), "closed training properties"
    print "There are ", len(set(closed_train_property_labels_threshold)), "closed training properties w/ threshold"

    '''
    Cost sensitive classification
    '''
    cost_matrices = {}

    for x in range(1, 3):
        # cost_matrices["open_cost_{0}".format(x)] = {'cost_matrix': []}
        # cost_matrices["closed_cost_{0}".format(x)] = {'cost_matrix': []}
        cost_matrices["threshold_open_cost_{0}".format(x)] = {'cost_matrix': []}
        cost_matrices["threshold_closed_cost_{0}".format(x)] = {'cost_matrix': []}
        # cost_matrices["costThreshold_open_cost_{0}".format(x)] = {'cost_matrix': []}
        # cost_matrices["costThreshold_closed_cost_{0}".format(x)] = {'cost_matrix': []}
        cost_matrices["thresholdCostThreshold_open_cost_{0}".format(x)] = {'cost_matrix': []}
        cost_matrices["thresholdCostThreshold_closed_cost_{0}".format(x)] = {'cost_matrix': []}

    model_path = os.path.join(sys.argv[4] + '/models.txt')

    with open(model_path, "wb") as out:
        json.dump(cost_matrices, out, indent=4)

    openKeySet = set(train_property_labels)
    openMapping = {val: key + 1 for key, val in enumerate(list(openKeySet))}
    openInvMapping = {key + 1: val for key, val in enumerate(list(openKeySet))}
    open_mapping_path = os.path.join(sys.argv[4] + '/open_label_mapping.txt')
    with open(open_mapping_path, "wb") as out:
        json.dump(openInvMapping, out, indent=4)

    openKeySetThreshold = copy.deepcopy(openKeySet)
    openKeySetThreshold |= {"no_region"}
    openMappingThreshold = {val: key + 1 for key, val in enumerate(list(openKeySetThreshold))}
    openInvMappingThreshold = {key + 1: val for key, val in enumerate(list(openKeySetThreshold))}
    open_mapping_path_threshold = os.path.join(sys.argv[4] + '/open_label_mapping_threshold.txt')
    with open(open_mapping_path_threshold, "wb") as out:
        json.dump(openInvMappingThreshold, out, indent=4)

    closedKeySet = set(closed_train_property_labels)
    closedMapping = {val: key + 1 for key, val in enumerate(list(closedKeySet))}
    closedInvMapping = {key + 1: val for key, val in enumerate(list(closedKeySet))}
    closed_mapping_path = os.path.join(sys.argv[4] + '/closed_label_mapping.txt')
    with open(closed_mapping_path, "wb") as out:
        json.dump(closedInvMapping, out, indent=4)

    closedKeySetThreshold = copy.deepcopy(closedKeySet)
    closedKeySetThreshold |= {"no_region"}
    closedMappingThreshold = {val: key + 1 for key, val in enumerate(list(closedKeySetThreshold))}
    closedInvMappingThreshold = {key + 1: val for key, val in enumerate(list(closedKeySetThreshold))}
    closed_mapping_path_threshold = os.path.join(sys.argv[4] + '/closed_label_mapping_threshold.txt')
    with open(closed_mapping_path_threshold, "wb") as out:
        json.dump(closedInvMappingThreshold, out, indent=4)

    print "Generating cost matrices...\n"
    genCostMatrices(pattern2regions, cost_matrices, openMapping, closedMapping, openMappingThreshold,
                    closedMappingThreshold, costThreshold, bias, slope)

    for i, value in enumerate(train_property_labels):
        train_property_labels[i] = openMapping[value]

    for i, value in enumerate(closed_train_property_labels):
        closed_train_property_labels[i] = closedMapping[value]


    '''
    Convert the labels to number not text format in a new matrix (final).
    '''
    final_cost_matrices = {}

    for x in range(1, 3):
        # final_cost_matrices["open_cost_{0}".format(x)] = {'cost_matrix': []}
        # final_cost_matrices["closed_cost_{0}".format(x)] = {'cost_matrix': []}
        final_cost_matrices["threshold_open_cost_{0}".format(x)] = {'cost_matrix': []}
        final_cost_matrices["threshold_closed_cost_{0}".format(x)] = {'cost_matrix': []}
        # final_cost_matrices["costThreshold_open_cost_{0}".format(x)] = {'cost_matrix': []}
        # final_cost_matrices["costThreshold_closed_cost_{0}".format(x)] = {'cost_matrix': []}
        final_cost_matrices["thresholdCostThreshold_open_cost_{0}".format(x)] = {'cost_matrix': []}
        final_cost_matrices["thresholdCostThreshold_closed_cost_{0}".format(x)] = {'cost_matrix': []}

    # These are all the file paths for all the models
    for model,dict in final_cost_matrices.items():
        dict['wordgrams_predict_path'] = os.path.join(sys.argv[5]+model+"_"+ str(threshold)+"_"+str(costThreshold)+"_"+str(bias)+"_"+str(slope)+"_wordgrams.predict")
        dict['wordgrams_prob_path'] = os.path.join(sys.argv[6]+model+"_"+ str(threshold)+"_"+str(costThreshold)+"_"+str(bias)+"_"+str(slope)+"_wordgrams.probpredict")

    for (model, arr),(finalModel, finalDict) in zip(cost_matrices.items(),final_cost_matrices.items()):
        if model.startswith("threshold") or model.startswith("thresholdCostThreshold"):
            if str(model).split('_')[1] == "open":
                for dict in arr['cost_matrix']:
                    tempDict = {}
                    for key,value in dict.items():
                        # print "Key is ",key, "value is", value
                        # print "Key is ",openMappingThreshold[key]
                        tempDict[openMappingThreshold[key]] = value
                        # print "tempDict",tempDict
                    finalDict['cost_matrix'].append(tempDict)
                        # dict[openMappingThreshold[key]] = dict.pop(key)
            else:
                for dict in arr['cost_matrix']:
                    tempDict = {}
                    for key,value in dict.items():
                        # print "Key is ",key, "value is", value
                        tempDict[closedMappingThreshold[key]] = value
                        # print "tempDict",tempDict
                    finalDict['cost_matrix'].append(tempDict)
                        # dict[closedMappingThreshold[key]] = dict.pop(key)
        else:
            if str(model).split('_')[1]=="open" or str(model).split('_')[0] == "open":
                for dict in arr['cost_matrix']:

                    tempDict = {}
                    for key,value in dict.items():
                        # print "Key is ",key, "value is", value
                        tempDict[openMapping[key]] = value
                        # print "tempDict",tempDict
                    finalDict['cost_matrix'].append(tempDict)
                        # dict[openMapping[key]] = dict.pop(key)
            else:
                for dict in arr['cost_matrix']:

                    tempDict = {}
                    for key,value in dict.items():
                        # print "Key is ",key, "value is", value
                        tempDict[closedMapping[key]] = value
                        # print "tempDict",tempDict
                    finalDict['cost_matrix'].append(tempDict)

    '''
    Generate two forms of prediction file
    '''

    print "Generating AROW predictions...\n"

    generatePredictions(final_cost_matrices, train_wordbigram_list,test_wordbigram_list)

