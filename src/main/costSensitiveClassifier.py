import math
import re
from nltk.corpus import stopwords  # Import the stop word list
import json
import numpy as np
import pandas as pd
from sklearn import preprocessing
import subprocess
import sys
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import os

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def sentence_to_words(sentence, remove_stopwords=False):
    letters_only = re.sub('[^a-zA-Z| LOCATION_SLOT | NUMBER_SLOT]', " ", sentence)
    # [x.strip() for x in re.findall('\s*(\w+|\W+)', line)]
    words = letters_only.lower().split()
    # In Python, searching a set is much faster than searching a list, so convert the stop words to a set
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return (words)

def genCostMatrices(inputSentences, costMatrix,openVector, closedVector):
    for i, sentence in enumerate(inputSentences[:15000]):
        if sentence:
            # Generate all the variables I need
            error = float(sentence['meanAbsError'])
            closedError = float(sentence['closedMeanAbsError'])

            openDict = sentence['openCostDict']
            closedDict = sentence['closedCostDict']

            openPrediction = sentence['predictedPropertyOpen']
            closedPrediction = sentence['predictedPropertyClosed']

            openCostVectorRem = np.sort([value for i,value in enumerate(sentence['openCostArr']) if value is not sentence['meanAbsError']])
            closedCostVectorRem = np.sort([value for i,value in enumerate(sentence['closedCostArr']) if value is not sentence['closedMeanAbsError']])
            # print "Closed cost array is",sentence['closedCostArr']
            # print "Open cost array is",sentence['openCostArr']

            openIQR = float(np.subtract(*np.percentile(openCostVectorRem, [75, 25])))
            closedIQR = float(np.subtract(*np.percentile(closedCostVectorRem, [75, 25])))
            openMedian = float(np.median(openCostVectorRem))
            closedMedian = float(np.median(closedCostVectorRem))
            openGap = float(sentence['openCostArr'][1]-sentence['meanAbsError'])
            # print "Open gap is",openGap
            closedGap = float(sentence['closedCostArr'][1]-sentence['closedMeanAbsError'])
            # print "Closed gap is",closedGap

            closedRange = float(np.ptp(sentence['closedCostArr']))
            # print "Closed range is",closedRange

            openRange = float(np.ptp(sentence['openCostArr']))
            # print "Open range is",openRange
            closedPercentArray = [float(val)/closedRange for val in sentence['closedCostArr']]
            openPercentArray = [float(val)/openRange for val in sentence['openCostArr']]

            # TODO - this is a hyperparameter
            closedCompetingEntries = float(sum(float(i) < float(1) for i in sentence['closedCostArr']))

            # print "Closed competing entries are",closedCompetingEntries

            openCompetingEntries = float(sum(float(i) < float(1) for i in sentence['openCostArr']))
            # print "openCompetingEntries are",openCompetingEntries

            closedGapPercent = float(closedGap/closedRange)
            # print "closedGapPercent is ",closedGapPercent

            openGapPercent = float(openGap/openRange)
            # print "openGapPercent is ",openGapPercent

            # TODO - can squeeze to the max value there
            ybins = [0,0.25,0.5,0.75,1]
            squeezeClosedArray = np.digitize(closedPercentArray, ybins, right=False)
            closedFactor = float(squeezeClosedArray.tolist().count(np.amin(squeezeClosedArray)))
            # print "Squeezed closed array is ",squeezeClosedArray
            # print "closedFactor is",closedFactor

            squeezeOpenArray = np.digitize(openPercentArray, ybins, right=False)
            openFactor = float(squeezeOpenArray.tolist().count(np.amin(squeezeOpenArray)))
            # print "Squeezed open array is ",squeezeOpenArray
            # print "openFactor is",openFactor

            # calculate the proportional values of samples
            # p = 1. * np.arange(len(sentence['openCostArr'])) / (len(sentence['openCostArr']) - 1)
            # print "array is ",sentence['openCostArr']
            # print "p is ",p
                        #
            # # plot the sort ed data:
            # fig = plt.figure()
            # ax1 = fig.add_subplot(121)
            # ax1.plot(p, sentence['openCostArr'])
            # ax1.set_xlabel('$p$')
            # ax1.set_ylabel('$x$')
            #
            # ax2 = fig.add_subplot(122)
            # ax2.plot(sentence['openCostArr'], p)
            # ax2.set_xlabel('$x$')
            # ax2.set_ylabel('$p$')
            # fig.show()
            # _ = raw_input("Press [enter] to continue.")

            for key, array in costMatrix.iteritems():
                # print key, value
                if str(key).startswith("single"):
                    if str(key).split('_')[1]=="open":
                        if str(key).endswith("1"):
                            array.append(error)
                        if str(key).endswith("2"):
                            array.append(error/float(openIQR*(len(openCostVectorRem)+1)))
                        if str(key).endswith("3"):
                            array.append(error/float(openMedian*(len(openCostVectorRem)+1)))
                        if str(key).endswith("4"):
                            array.append(error/np.where(openGap
    >0,openGap, 0e-10))
                        if str(key).endswith("5"):
                            array.append(error/np.where(openGapPercent
    >0,openGapPercent, 0e-10))
                        if str(key).endswith("6"):
                            array.append(error*openCompetingEntries)
                        if str(key).endswith("7"):
                            array.append(error*openFactor)
                        if str(key).endswith("8"):
                            array.append(closedError*closedFactor)
                    else:
                        if str(key).endswith("1"):
                            array.append(closedError)
                        if str(key).endswith("2"):
                            array.append(closedError/float(closedIQR*(len(closedCostVectorRem)+1)))
                        if str(key).endswith("3"):
                            array.append(closedError/float(closedMedian*(len(closedCostVectorRem)+1)))
                        if str(key).endswith("4"):
                            array.append(closedError/np.where(closedGap>0,closedGap,0e-10))
                        if str(key).endswith("5"):
                            array.append(closedError/np.where(closedGapPercent>0,closedGapPercent,0e-10))
                        if str(key).endswith("6"):
                            array.append(closedError*closedCompetingEntries)
                        if str(key).endswith("7"):
                            array.append(closedError*closedFactor)
                        if str(key).endswith("8"):
                            array.append(closedError*closedFactor)
                else:
                    if str(key).split('_')[0]=="open":
                        if str(key).endswith("1"):
                            # dict = {}
                            # for key,value in openDict.iteritems():
                            #     dict[key] = value/openRange
                            # array.append(dict)
                            # array.append(openDict)
                            # dict = {}
                            # for key,value in openDict.iteritems():
                            #     if key == openPrediction:
                            #         dict[key] = 0
                            #     else:
                            #         dict[key] = 1
                            # array.append(dict)
                            dict = {}
                            for i,key in enumerate(openKeySet):
                                if key == openPrediction:
                                    dict[key] = 0
                                else:
                                    dict[key] = 1
                            array.append(dict)
                        if str(key).endswith("2"):
                            dict = {}
                            for key,value in openDict.iteritems():
                                dict[key] = value/float(openIQR*(len(openCostVectorRem)+1))
                            array.append(dict)
                        if str(key).endswith("3"):
                            dict = {}
                            for key,value in openDict.iteritems():
                                dict[key] = value/float(openMedian*(len(openCostVectorRem)+1))
                            array.append(dict)
                        if str(key).endswith("4"):
                            dict = {}
                            for key,value in openDict.iteritems():
                                dict[key] = value/np.where(openGap
    >0,openGap, 0e-10)
                            array.append(dict)
                        if str(key).endswith("5"):
                            dict = {}
                            for key,value in openDict.iteritems():
                                dict[key] = value/np.where(openGapPercent
    >0,openGapPercent, 0e-10)
                            array.append(dict)
                        if str(key).endswith("6"):
                            dict = {}
                            for key,value in openDict.iteritems():
                                dict[key] = value*openCompetingEntries
                            array.append(dict)
                        if str(key).endswith("7"):
                            dict = {}
                            for key,value in openDict.iteritems():
                                dict[key] = value*openFactor
                            array.append(dict)
                        if str(key).endswith("7"):
                            dict = {}
                            for key,value in openDict.iteritems():
                                dict[key] = value*openFactor
                            array.append(dict)
                        if str(key).endswith("8"):
                            dict = {}
                            for key,value in openDict.iteritems():
                                if key == openPrediction:
                                    dict[key] = 0
                                else:
                                    dict[key] = 1
                            array.append(dict)
                    else:
                        if str(key).endswith("1"):
                            # dict = {}
                            # for key,value in closedDict.iteritems():
                            #     if key == closedPrediction:
                            #         dict[key] = 0
                            #     else:
                            #         dict[key] = 1
                            # array.append(dict)
                            dict = {}
                            for i,key in enumerate(closedKeySet):
                                if key == closedPrediction:
                                    dict[key] = 0
                                else:
                                    dict[key] = 1
                            array.append(dict)
                            # dict = {}
                            # for key,value in closedDict.iteritems():
                            #     dict[key] = value/closedRange
                            # array.append(dict)
                            # array.append(closedDict)
                        if str(key).endswith("2"):
                            dict = {}
                            for key,value in closedDict.iteritems():
                                dict[key] = value/float(closedIQR*(len(closedCostVectorRem)+1))
                            array.append(dict)
                        if str(key).endswith("3"):
                            dict = {}
                            for key,value in closedDict.iteritems():
                                dict[key] = value/float(closedMedian*(len(closedCostVectorRem)+1))
                            array.append(dict)
                        if str(key).endswith("4"):
                            dict = {}
                            for key,value in closedDict.iteritems():
                                dict[key] = value/np.where(closedGap>0,closedGap,0e-10)
                            array.append(dict)
                        if str(key).endswith("5"):
                            dict = {}
                            for key,value in closedDict.iteritems():
                                dict[key] = value/np.where(closedGapPercent>0,closedGapPercent,0e-10)
                            array.append(dict)
                        if str(key).endswith("6"):
                            dict = {}
                            for key,value in closedDict.iteritems():
                                dict[key] = value*closedCompetingEntries
                            array.append(dict)
                        if str(key).endswith("7"):
                            dict = {}
                            for key,value in closedDict.iteritems():
                                dict[key] = value*closedFactor
                            array.append(dict)
                        if str(key).endswith("8"):
                            dict = {}
                            for key,value in closedDict.iteritems():
                                if key == closedPrediction:
                                    dict[key] = 0
                                else:
                                    dict[key] = 1
                            array.append(dict)

    # for key, value in costMatrix2.iteritems():
    #     print key," array is ", value,"\n"

def generateDatFiles(costDict, trainingLabels, closedTrainingLabels, trainingFeatures, pathDict):

    # TODO - make sure this opens a fresh file all the time and replaces

    print "There are this many unique labels",len(set(trainingLabels))
    print "There are this many closed unique labels",len(set(closedTrainingLabels))

    print "Generating VW files...\n"

    for (model,filepath), (model,costArray) in zip(pathDict.items(), costDict.items()):
        f = open(filepath, 'w')
        for i, (label,closedLabel,costDict, features) in enumerate(zip(trainingLabels,closedTrainingLabels,costArray,trainingFeatures)):
            line = ""
            if model.startswith("single"):
                if str(model).split('_')[1]=="open":
                    # features for vowpal format only
                    # + " " + str(i) +
                    #  (" ").join(map(str, features))
                    line += str(label) + ":" + str(costDict) + " | " + features
                    f.write(line+"\n")
                else:
                    line += str(closedLabel) + ":" + str(costDict) + " | " + features
                    f.write(line+"\n")
            else:
                for lb, cost in costDict.items():
                     # + " "
                    line += str(lb) + ":" + str(cost) + " "
                # str(i) +
                line +=  "| " + features
                f.write(line+"\n")


def training_features(inputSentences):
    global vectorizer

    for i, sentence in enumerate(inputSentences[:15000]):
        # Dont train if the sentence contains a random region we don't care about
        # and sentence['predictedRegion'] in properties
        if sentence:
            # Regardless of anything else, an open evaluation includes all sentences
            train_wordlist.append(" ".join(sentence_to_words(sentence['parsedSentence'], True)))
            train_property_labels.append(sentence['predictedPropertyOpen'])
            train_property_labels_threshold.append(sentence['predictedPropertyOpenThreshold'])
            # Closed evaluation only include certain training sentences
            closed_train_property_labels.append(sentence['predictedPropertyClosed'])
            closed_train_property_labels_threshold.append(sentence['predictedPropertyClosedThreshold'])

    # print "These are the clean words in the training sentences: ", train_wordlist
    # print "These are the labels in the training sentences: ", train_labels
    print "Creating the bag of words...\n"
    train_data_features = train_wordlist
    # train_data_features = vectorizer.fit_transform(train_wordlist)
    # train_data_features = train_data_features.toarray()
    # train_data_features = train_data_features.astype(np.float)

    return train_data_features

def test_features(testSentences):
    global vectorizer
    global cost_mat_test

    for sentence in testSentences:
        if sentence['parsedSentence'] != {} and sentence['mape_label'] != {}:
            clean_test_sentences.append(" ".join(sentence_to_words(sentence['parsedSentence'], True)))
            test_property_labels.append(sentence['property'])
    # print "These are the clean words in the test sentences: ", clean_test_sentences
    # print "These are the mape labels in the test sentences: ", binary_test_labels
    test_data_features = clean_test_sentences
    # test_data_features = vectorizer.transform(clean_test_sentences)
    # test_data_features = test_data_features.toarray()
    # test_data_features = test_data_features.astype(np.float)

    return test_data_features

if __name__ == "__main__":
    # training data
    # load the sentence file for training

    with open(sys.argv[1]) as trainingSentences:
        pattern2regions = json.loads(trainingSentences.read())

    print "We have ", len(pattern2regions), " training sentences."
    # We load in the allowable features and also no_region
    with open(sys.argv[3]) as featuresKept:
        properties = json.loads(featuresKept.read())
    # properties.append("no_region")

    with open(sys.argv[2]) as testSentences:
        testSentences = json.loads(testSentences.read())

    finalTestSentences = []

    for sentence in testSentences:
        finalTestSentences.append(sentence)

    print "Length of final sentences is", len(finalTestSentences)

    train_wordlist = []
    closed_train_wordlist = []
    test_wordlist = []
    train_property_labels = []
    train_property_labels_threshold = []
    closed_train_property_labels = []
    closed_train_property_labels_threshold = []
    test_property_labels = []



    vectorizer = CountVectorizer(analyzer="word", \
                                 tokenizer=None, \
                                 preprocessor=None, \
                                 stop_words=None, \
                                 max_features=5000)

    print "Getting all the words in the training sentences...\n"

    # This both sorts out the features and the training labels
    train_data_features = training_features(pattern2regions)



    print len(train_data_features), "sets of training features"

    #
    trainingLabels = len(train_data_features)
    positiveOpenTrainingLabels = len(train_property_labels_threshold) - train_property_labels_threshold.count(
        "no_region")
    negativeOpenTrainingLabels = train_property_labels_threshold.count("no_region")
    positiveClosedTrainingLabels = len(closed_train_property_labels_threshold) - closed_train_property_labels_threshold.count(
        "no_region")
    negativeClosedTrainingLabels = closed_train_property_labels_threshold.count(
        "no_region")
    #
    #
    # print "There are ", len(train_property_labels_threshold) - train_property_labels_threshold.count(
    #     "no_region"), "positive open mape threshold labels"
    # print "There are ", train_property_labels_threshold.count("no_region"), "negative open mape threshold labels"
    # print "There are ", len(closed_train_property_labels_threshold) - closed_train_property_labels_threshold.count(
    #     "no_region"), "positive closed mape threshold labels"
    # print "There are ", closed_train_property_labels_threshold.count(
    #     "no_region"), "negative closed mape threshold labels\n"

    # Fit the logistic classifiers to the training set, using the bag of words as features

    openTrainingClasses = len(set(train_property_labels))
    openTrainingClassesThreshold = len(set(train_property_labels_threshold))
    closedTrainingClasses = len(set(closed_train_property_labels))
    closedTrainingClassesThreshold = len(set(closed_train_property_labels_threshold))


    print "There are ", len(set(train_property_labels)), "open training classes"
    print "There are ", len(set(train_property_labels_threshold)), "open training classes w/threshold"
    print "There are ", len(set(closed_train_property_labels)), "closed training properties"
    print "There are ", len(set(closed_train_property_labels_threshold)), "closed training properties w/ threshold"

    # Create an empty list and append the clean reviews one by one
    clean_test_sentences = []

    print "Cleaning and parsing the test set ...\n"

    print "Get a bag of words for the test set, and convert to a numpy array\n"

    test_data_features = test_features(finalTestSentences)

    print len(test_data_features), "sets of test features"

    '''
    Cost sensitive classification
    '''

    cost_matrices = {}

    for x in range(1,2):
        cost_matrices["open_cost_{0}".format(x)]=[]
        cost_matrices["closed_cost_{0}".format(x)]=[]
        # cost_matrices["single_open_cost_{0}".format(x)]=[]
        # cost_matrices["single_closed_cost_{0}".format(x)]=[]

    model_path = os.path.join(sys.argv[4] +'/models.txt')

    with open(model_path, "wb") as out:
        json.dump(cost_matrices, out,indent=4)

    openKeySet = set(train_property_labels)
    openMapping = {val:key+1 for key,val in enumerate(list(openKeySet))}
    openInvMapping = {key+1:val for key,val in enumerate(list(openKeySet))}

    open_mapping_path = os.path.join(sys.argv[4] +'/open_label_mapping.txt')

    with open(open_mapping_path, "wb") as out:
        json.dump(openInvMapping, out,indent=4)

    closedKeySet = set(closed_train_property_labels)
    closedMapping = {val:key+1 for key,val in enumerate(list(closedKeySet))}
    closedInvMapping = {key+1:val for key,val in enumerate(list(closedKeySet))}

    closed_mapping_path = os.path.join(sys.argv[4] +'/closed_label_mapping.txt')

    with open(closed_mapping_path, "wb") as out:
        json.dump(closedInvMapping, out,indent=4)

    genCostMatrices(pattern2regions,cost_matrices,openKeySet,closedKeySet)

    paths = {}

    def generateDatPaths(model):
        paths[model]=os.path.join(sys.argv[4]+model+".dat")

    for key in cost_matrices.keys():
        generateDatPaths(key)

    # This has to be done after we gain the mappings
    # openle = preprocessing.LabelEncoder()
    # closedle = preprocessing.LabelEncoder()
    #
    # openle.fit(train_property_labels)

    for i,value in enumerate(train_property_labels):
        train_property_labels[i]=openMapping[value]

    # train_property_labels = openle.transform(train_property_labels)
    # print "Single training labels are ",train_property_labels

    for i,value in enumerate(closed_train_property_labels):
        closed_train_property_labels[i]=closedMapping[value]

    # closedle.fit(closed_train_property_labels)
    # closed_train_property_labels = closedle.transform(closed_train_property_labels)
    # print "Single training labels are ",closed_train_property_labels

    for model,arr in cost_matrices.items():
        if not model.startswith("single"):
            if str(model).split('_')[0]=="open":
                for open_dict in arr:
                    # print "Open dict is", open_dict
                    for key in open_dict.keys():
                        # print "Key is",key
                        open_dict[openMapping[key]] = open_dict.pop(key)
            else:
                for inner_dict in arr:
                    # print "Closed dict is", inner_dict
                    # print "Model is ",model
                    for key in inner_dict.keys():
                        # print "Key is",key
                        inner_dict[closedMapping[key]] = inner_dict.pop(key)


    generateDatFiles(cost_matrices,train_property_labels,closed_train_property_labels,train_data_features, paths)

    test_path = os.path.join(sys.argv[4]+"test.dat")

    # Generate test file
    testfile = open(test_path, 'w')
    for i,features in enumerate(test_data_features):
        # str(i) + "| " +
        line = features
        testfile.write(line+"\n")

    # train_commands = {}
    # predict_commands = {}
    #
    # def genVWCommands(model):
    #     if str(model).startswith("single"):
    #         if str(model).split('_')[1]=="open":
    #             train_commands[model]=("vw --csoaa 24 " + os.path.join(sys.argv[4]+model+".dat") +" -f "+os.path.join(sys.argv[4])+model+".model")
    #             # train_commands[model]=("vw --csoaa 24 " + os.path.join("data/cost/"+model+".dat") +" -f "+os.path.join("data/cost/")+model+".model")
    #         else:
    #             train_commands[model]=("vw --csoaa 16 " + os.path.join(sys.argv[4]+model+".dat") +" -f "+os.path.join(sys.argv[4])+model+".model")
    #             # train_commands[model]=("vw --csoaa 16 " + os.path.join("data/cost/"+model+".dat") +" -f "+os.path.join("data/cost/")+model+".model")
    #     else:
    #         if str(model).split('_')[0]=="open":
    #             train_commands[model]=("vw --csoaa 24 " + os.path.join(sys.argv[4]+model+".dat") +" -f "+os.path.join(sys.argv[4])+model+".model")
    #             # train_commands[model]=("vw --csoaa 24 " + os.path.join("data/cost/"+model+".dat") +" -f "+os.path.join("data/cost/")+model+".model")
    #         else:
    #             train_commands[model]=("vw --csoaa 16 " + os.path.join(sys.argv[4]+model+".dat") +" -f "+os.path.join(sys.argv[4])+model+".model")
    #             # train_commands[model]=("vw --csoaa 16 " + os.path.join("data/cost/"+model+".dat") +" -f "+os.path.join("data/cost/")+model+".model")
    #     predict_commands[model]=("vw -t -i "+os.path.join(sys.argv[4])+model+".model " + os.path.join(sys.argv[4]+"test.dat")+" -p "+os.path.join(sys.argv[4])+model+ ".predict")
    #     # predict_commands[model]=("vw -t -i "+os.path.join("data/cost/")+model+".model " + os.path.join("data/cost/"+"test.dat")+" -p "+os.path.join("data/cost/")+model+ ".predict")
    #
    # for key in cost_matrices.keys():
    #     genVWCommands(key)
    #
    # print train_commands
    # print predict_commands

    # for (model,trainCommand), (model,predictCommand) in zip(train_commands.items(), predict_commands.items()):
    #     print "Model is",model
    #     print "Training command is ",trainCommand
    #     print "predictCommand command is ",predictCommand
    #     subprocess.call(trainCommand.split(' '))
    #     # p1.wait()
    #     print "Finished Training: ", model
    #     subprocess.call(predictCommand.split(' '))
    #     # p2.wait()
    #     print "Finished Predicting: ",model
    #
    # cost_predictions = {}
    #
    # for model in cost_matrices.keys():
    #     print "Model is",model
    #     cost_predictions[model]=np.loadtxt(os.path.join(sys.argv[4]+model+".predict"))
    #     cost_predictions[model] = map(int,[i[0] for i in cost_predictions[model]])
    #     print "Predictions are",len(cost_predictions[model])
    #     # print "Mapped predictions are",openle.inverse_transform(cost_predictions[model])
    #     if str(model).startswith("single"):
    #         if str(model).split('_')[1]=="open":
    #             cost_predictions[model]=[openInvMapping[i] for i in cost_predictions[model]]
    #         else:
    #             cost_predictions[model]=[closedInvMapping[i] for i in cost_predictions[model]]
    #     else:
    #         if str(model).split('_')[0]=="open":
    #             cost_predictions[model]=[openInvMapping[i] for i in cost_predictions[model]]
    #         else:
    #             cost_predictions[model]=[closedInvMapping[i] for i in cost_predictions[model]]
    #
    #
    # print cost_predictions
    #
    # #
    # # Load in the test data
    # test = pd.DataFrame(finalTestSentences)
    # threshold = test['threshold'][0]
    #
    # # These are the ground truths
    #
    # y_multi_true = np.array(test['property'])
    # y_true_claim = np.array(test['claim'])
    #
    # binary_cslr_predictions = {}
    #
    # for key,predictions in cost_predictions.iteritems():
    #     if predictions:
    #         binary_cslr_predictions[key]=[]
    #         for predict,true in zip(predictions,y_multi_true):
    #             if predict==true:
    #                 binary_cslr_predictions[key].append(1)
    #             else:
    #                 binary_cslr_predictions[key].append(0)
    #
    # print binary_cslr_predictions
    #
    # # TODO This was an issue on command line - change to [2] if on command line and 8 if not
    # testSet = str(os.path.splitext(sys.argv[2])[0]).split("/")[8]
    #
    # # # # TODO - need to create a per property chart
    # # Now we write our precision F1 etc to an Excel file
    # summaryDF = pd.DataFrame(columns=('precision', 'recall', 'f1', 'accuracy', 'evaluation set', 'threshold'))
    # # # 'probThreshold'
    # #
    # def evaluation(trueLabels, evalLabels, test_set, threshold):
    #     # ,probThreshold
    #     global summaryDF
    #     global trainingLabels
    #     global positiveOpenTrainingLabels
    #     global negativeOpenTrainingLabels
    #     global positiveClosedTrainingLabels
    #     global negativeClosedTrainingLabels
    #     global openTrainingClasses
    #     global openTrainingClassesThreshold
    #     global closedTrainingClasses
    #     global closedTrainingClassesThreshold
    #
    #     precision = precision_score(trueLabels, evalLabels)
    #     recall = recall_score(trueLabels, evalLabels)
    #     f1 = f1_score(trueLabels, evalLabels)
    #     accuracy = accuracy_score(trueLabels, evalLabels)
    #
    #     data = {'precision': [precision],
    #             'recall': [recall],
    #             'f1': [f1],
    #             'accuracy': [accuracy],
    #             'evaluation set': [test_set],
    #             'threshold': [threshold],
    #             # 'probThreshold': [probThreshold],
    #             'trainingLabels':[trainingLabels],
    #             'positiveOpenLabels':[positiveOpenTrainingLabels],
    #             'negativeOpenLabels':[negativeOpenTrainingLabels],
    #             'positiveClosedLabels':[positiveClosedTrainingLabels],
    #             'negativeClosedLabels':[negativeClosedTrainingLabels],
    #             'openTrainingClasses': [openTrainingClasses],
    #             'openTrainingClassesThreshold': [openTrainingClassesThreshold],
    #             'closedTrainingClasses':[closedTrainingClasses],
    #             'closedTrainingClassesThreshold': [closedTrainingClassesThreshold]
    #             }
    #
    #     DF = pd.DataFrame(data)
    #
    #     summaryDF = pd.concat([summaryDF, DF])
    #
    # for model,result in binary_cslr_predictions.iteritems():
    #     print "Model is ",model
    #     print "True claim is",np.array(y_true_claim)
    #     print "Result is",result
    #
    #     evaluation(y_true_claim, result, testSet, threshold)
    #
    # columns = list(binary_cslr_predictions.keys())
    #
    # summaryDF.index = columns
    #
    # print summaryDF
    # #
    # try:
    #     if os.stat(sys.argv[5]).st_size > 0:
    #         # df_csv = pd.read_csv(sys.argv[5],encoding='utf-8',engine='python')
    #         # summaryDF = pd.concat([df_csv,summaryDF],axis=1,ignore_index=True)
    #         with open(sys.argv[5], 'a') as f:
    #             # Need to empty file contents now
    #             summaryDF.to_csv(path_or_buf=f, encoding='utf-8', mode='a', header=False)
    #             f.close()
    #     else:
    #         print "empty file"
    #         with open(sys.argv[5], 'w+') as f:
    #             summaryDF.to_csv(path_or_buf=f, encoding='utf-8')
    #             f.close()
    # except OSError:
    #     print "No file"
    #     with open(sys.argv[5], 'w+') as f:
    #         summaryDF.to_csv(path_or_buf=f, encoding='utf-8')
    #         f.close()