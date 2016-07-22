'''

This file predicts only once for models that do not change with hyperparameters:

Closed_Property_Bag_of_Words_Multinomial_Logistic_Regression_w_Binary_Evaluation
Open_Property_Bag_of_Words_Multinomial_Logistic_Regression_w_Binary_Evaluation

Open_Property_Distant_Supervision_Model
Closed_Property_Distant_Supervision_Model

Random Binary Baseline

Open_Categorical_Random_Baseline
Open_Categorical_Random Baseline_w_Threshold
Closed_Categorical_Random Baseline
Closed_Categorical_Random Baseline_w_Threshold

Negative_Naive_Baseline

Previous_Model

'''

import re
from nltk.corpus import stopwords  # Import the stop word list
import json
import numpy as np
import pandas as pd
import sys
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import os
from itertools import repeat
import copy

rng = np.random.RandomState(101)

#
# python src/main/trainingFeatures.py data/output/predictedProperties.json data/output/hyperTestLabels.json data/regressionResult.json

'''TODO -
    Bi-grams, LSTMs, Word2Vec
    Cost-sensitive classification
    Use class-weight = balanced
    Cross validation
    Check training parameters and that they have been parsed correctly
    Precision, recall, F1 for each region
    Are we training on too many positive instances (no region)?
'''

def sentence_to_words(sentence, remove_stopwords=False):
    letters_only = re.sub('[^a-zA-Z| LOCATION_SLOT | NUMBER_SLOT]', " ", sentence)
    # [x.strip() for x in re.findall('\s*(\w+|\W+)', line)]
    words = letters_only.lower().split()

    # In Python, searching a set is much faster than searching a list, so convert the stop words to a set
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return (words)


def training_features(inputSentences):
    global vectorizer
    # [:15000]
    for i, sentence in enumerate(inputSentences[:15000]):
        # Dont train if the sentence contains a random region we don't care about
        # and sentence['predictedRegion'] in properties
        if sentence:
            # Regardless of anything else, an open evaluation includes all sentences
            train_wordlist.append(" ".join(sentence_to_words(sentence['parsedSentence'], True)))
            train_property_labels.append(sentence['predictedPropertyOpen'])
            # Closed evaluation only include certain training sentences
            closed_train_property_labels.append(sentence['predictedPropertyClosed'])
    # print "These are the clean words in the training sentences: ", train_wordlist
    # print "These are the labels in the training sentences: ", train_labels
    print "Creating the bag of words...\n"
    train_data_features = vectorizer.fit_transform(train_wordlist)
    train_data_features = train_data_features.toarray()

    return train_data_features


def test_features(testSentences):
    global vectorizer

    for sentence in testSentences:
        if sentence['parsedSentence'] != {} and sentence['mape_label'] != {}:
            clean_test_sentences.append(" ".join(sentence_to_words(sentence['parsedSentence'], True)))
            test_property_labels.append(sentence['property'])

    # print "These are the clean words in the test sentences: ", clean_test_sentences
    # print "These are the mape labels in the test sentences: ", binary_test_labels
    test_data_features = vectorizer.transform(clean_test_sentences)
    test_data_features = test_data_features.toarray()

    return test_data_features

# Find index of the true label for the sentence, and if that same index for that sentence is one, return the classifier class, else no region

if __name__ == "__main__":
    # training data
    # load the sentence file for training

    with open(sys.argv[1]) as trainingSentences:
        pattern2regions = json.loads(trainingSentences.read())

    print "We have ", len(pattern2regions), " training sentences."
    # We load in the allowable features and also no_region
    with open(sys.argv[3]) as featuresKept:
        properties = json.loads(featuresKept.read())
    properties.append("no_region")

    with open(sys.argv[2]) as testSentences:
        testSentences = json.loads(testSentences.read())

    finalTestSentences = []

    for sentence in testSentences:
        if sentence['parsedSentence'] != {} and sentence['mape_label'] != {} and sentence['mape'] != {} and sentence[
            'property'] != {} and sentence['property'] in properties:
            # print sentence['property']
            finalTestSentences.append(sentence)

    train_wordlist = []
    closed_train_wordlist = []
    test_wordlist = []
    binary_test_labels = []
    train_property_labels = []
    closed_train_property_labels = []
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

    multi_logit = LogisticRegression(fit_intercept=True, class_weight='auto', multi_class='multinomial',
                                     solver='newton-cg')

    closed_multi_logit = LogisticRegression(fit_intercept=True, class_weight='auto', multi_class='multinomial',
                                            solver='newton-cg')

    # Fit the logistic classifiers to the training set, using the bag of words as features
    print "There are ", len(set(train_property_labels)), "open training classes"
    print "There are ", len(set(closed_train_property_labels)), "closed training properties"

    print "Fitting the open multinomial logistic regression model without MAPE threshold...\n"
    open_multi_logit = multi_logit.fit(train_data_features, train_property_labels)

    print "Fitting the closed multinomial logistic regression model without MAPE threshold...\n"
    closed_multi_logit = closed_multi_logit.fit(train_data_features, closed_train_property_labels)

    # Create an empty list and append the clean reviews one by one
    clean_test_sentences = []

    print "Cleaning and parsing the test set ...\n"

    print "Get a bag of words for the test set, and convert to a numpy array\n"

    test_data_features = test_features(finalTestSentences)

    print len(test_data_features), "sets of test features"

    print test_data_features
    #
    print "Predicting open multinomial test labels without MAPE threshold...\n"
    y_multi_logit_result_open = np.array(open_multi_logit.predict(test_data_features))

    print "Predicting closed multinomial test labels w/ threshold...\n"
    y_multi_logit_result_closed = np.array(closed_multi_logit.predict(test_data_features))

    # Load in the test data
    test = pd.DataFrame(finalTestSentences)

    # These are the ground truths
    y_multi_true = np.array(test['property'])
    y_true_claim = np.array(test['claim'])

    y_multi_logit_result_open_binary = []
    y_multi_logit_result_closed_binary = []

    # This is Andreas model for distant supervision
    y_andreas_mape = test['mape_label']

    y_distant_sv_property_open = test['predictedPropertyOpen']
    y_distant_sv_property_closed = test['predictedPropertyClosed']

    y_open_distant_sv_to_binary = []
    y_closed_distant_sv_to_binary = []

    # These are the random baselines
    unique_train_labels = set(train_property_labels)
    # print "Open property labels",unique_train_labels
    # unique_train_labels_threshold = copy.deepcopy(unique_train_labels)
    # unique_train_labels_threshold = unique_train_labels_threshold.add('no_region')
    unique_train_labels_threshold = unique_train_labels.union(['no_region'])
    # print "Open property label with threshold",unique_train_labels_threshold

    closed_unique_train_labels = set(closed_train_property_labels)
    # print type(closed_unique_train_labels)
    # closed_unique_train_labels_threshold = copy.deepcopy(closed_unique_train_labels)
    # closed_unique_train_labels_threshold = closed_unique_train_labels_threshold.add('no_region')
    closed_unique_train_labels_threshold = closed_unique_train_labels.union(['no_region'])

    # print "Closed property label threshold",type(closed_unique_train_labels_threshold)

    # Categorical random baseline
    categorical_random = rng.choice(list(unique_train_labels), len(finalTestSentences))
    categorical_random_threshold = rng.choice(list(unique_train_labels_threshold), len(finalTestSentences))
    closed_categorical_random = rng.choice(list(closed_unique_train_labels), len(finalTestSentences))
    closed_categorical_random_threshold = rng.choice(list(closed_unique_train_labels_threshold),len(finalTestSentences))
    # print "Categorical random is ", categorical_random
    y_cat_random_to_binary = []
    y_cat_random_to_binary_threshold = []
    y_closed_random_to_binary = []
    y_closedCat_random_to_binary_threshold = []
    # Random 0 and 1
    random_result = rng.randint(2, size=len(finalTestSentences))
    positive_result = np.ones(len(finalTestSentences))
    negative_result = np.zeros(len(finalTestSentences))
    y_randpred = random_result
    y_pospred = positive_result
    y_negpred = negative_result
    #
    catLabels = [y_distant_sv_property_open, y_distant_sv_property_closed,
                 y_multi_logit_result_open,
                 y_multi_logit_result_closed,
                 categorical_random,
                 categorical_random_threshold,
                 closed_categorical_random,
                 closed_categorical_random_threshold
                 ]
    #
    binaryLabels = [
        y_open_distant_sv_to_binary, y_closed_distant_sv_to_binary,
        y_multi_logit_result_open_binary,
        y_multi_logit_result_closed_binary,
        y_cat_random_to_binary, y_cat_random_to_binary_threshold, y_closed_random_to_binary,
        y_closedCat_random_to_binary_threshold
    ]
    #
    trueLabels = []
    trueLabels.extend(repeat(y_multi_true, len(binaryLabels)))
    #
    #
    # Convert the categorical predictions to binary based on if matching property
    def binaryConversion(trueLabel, evalLabel, binaryLabel):
        for true, eval in zip(trueLabel, evalLabel):
            # print true,eval, binary
            if eval == true:
                binaryLabel.append(1)
            else:
                binaryLabel.append(0)

                # print x,y,z
    #
    #
    for trueLabels, predictionLabels, emptyArray in zip(trueLabels, catLabels, binaryLabels):
        binaryConversion(trueLabels, predictionLabels, emptyArray)

    output = pd.DataFrame(data=dict(parsed_sentence=test['parsedSentence'],
                                    features=clean_test_sentences,

                                    open_property_prediction=y_multi_logit_result_open,
                                    open_property_prediction_toBinary=y_multi_logit_result_open_binary,
                                    closed_property_prediction=y_multi_logit_result_closed,
                                    closed_property_prediction_toBinary=y_multi_logit_result_closed_binary,

                                    distant_supervision_open=y_distant_sv_property_open,
                                    distant_supervision_closed=y_distant_sv_property_closed,
                                    distant_supervision_open_toBinary=y_open_distant_sv_to_binary,
                                    distant_supervision_closed_toBinary=y_closed_distant_sv_to_binary,

                                    random_binary_label=random_result,
                                    random_categorical_label=categorical_random,
                                    random_categorical_label_toBinary=y_cat_random_to_binary,
                                    random_categorical_label_threshold=categorical_random_threshold,
                                    random_categorical_label_threshold_toBinary=y_cat_random_to_binary_threshold,
                                    closed_random_categorical_label=closed_categorical_random,
                                    closed_random_categorical_label_toBinary=y_closed_random_to_binary,
                                    closed_random_categorical_label_threshold=closed_categorical_random_threshold,
                                    closed_random_categorical_label_toBinary_threshold=y_closedCat_random_to_binary_threshold,

                                    test_data_mape_label=test['mape_label'],
                                    claim_label=y_true_claim,
                                    test_data_property_label=test['property'],
                                    andreas_prediction=y_pospred,
                                    negative_baseline=y_negpred))

    # print str(os.path.splitext(sys.argv[2])[0]).split("/")
    # TODO This was an issue on command line - change to [2] if on command line, 8 if not
    testSet = str(os.path.splitext(sys.argv[2])[0]).split("/")[8]

    resultPath = os.path.join(sys.argv[4] + "test/" + testSet + '_mainRegressionResult.csv')

    output.to_csv(path_or_buf=resultPath, encoding='utf-8', index=False, cols=[
        'parsed_sentence',
        'features',
        'open_property_prediction',
        'open_property_prediction_toBinary',
        'closed_property_prediction',
        'closed_property_prediction_toBinary',

        'distant_supervision_open',
        'distant_supervision_closed',
        'distant_supervision_open_toBinary',
        'distant_supervision_closed_toBinary',

        'random_binary_label',
        'random_categorical_label',
        'random_categorical_label_toBinary',
        'random_categorical_label_threshold',
        'random_categorical_label_threshold_toBinary',
        'closed_random_categorical_label',
        'closed_random_categorical_label_toBinary',
        'closed_random_categorical_label_threshold',
        'closed_random_categorical_label_toBinary_threshold',

        'test_data_mape_label',
        'claim_label',
        'andreas_property_label',
        'andreas_prediction',
        'negative_baseline',
    ])

    # TODO - need to create a per property chart

    # Now we write our precision F1 etc to an Excel file
    summaryDF = pd.DataFrame(columns=('precision', 'recall', 'f1', 'accuracy', 'evaluation set'))


    def evaluation(trueLabels, evalLabels, test_set):
        global summaryDF

        precision = precision_score(trueLabels, evalLabels)
        recall = recall_score(trueLabels, evalLabels)
        f1 = f1_score(trueLabels, evalLabels)
        accuracy = accuracy_score(trueLabels, evalLabels)

        data = {'precision': [precision],
                'recall': [recall],
                'f1': [f1],
                'accuracy': [accuracy],
                'evaluation set': [test_set]
                }

        DF = pd.DataFrame(data)

        summaryDF = pd.concat([summaryDF, DF])


    results = [y_randpred,
               y_cat_random_to_binary,
               y_cat_random_to_binary_threshold,
               y_closed_random_to_binary,
               y_closedCat_random_to_binary_threshold,
               y_pospred,
               y_negpred,
               y_open_distant_sv_to_binary,
               y_closed_distant_sv_to_binary,
               y_multi_logit_result_open_binary,
               y_multi_logit_result_closed_binary,
               ]
    #

    for result in results:
        evaluation(y_true_claim, result, testSet)

    columns = list(['Binary_Random_Baseline',
                    'Open_Categorical_Random_Baseline',
                    'Open_Categorical_Random Baseline_w_Threshold',
                    'Closed_Categorical_Random Baseline',
                    'Closed_Categorical_Random Baseline_w_Threshold',
                    'Previous_Model',
                    'Negative_Naive_Baseline',
                    'Open_Property_Distant_Supervision_Model',
                    'Closed_Property_Distant_Supervision_Model',
                    'Open_Property_Bag_of_Words_Multinomial_Logistic_Regression_w_Binary_Evaluation',
                    'Closed_Property_Bag_of_Words_Multinomial_Logistic_Regression_w_Binary_Evaluation',
                    ])

    # summaryDF.set_index(['A','B'])
    summaryDF.index = columns

    print summaryDF
    #
    # precisionF1Path = os.path.join(sys.argv[4] + "test/" + testSet + '_' + str(threshold) + '_summaryEval.csv')
    # summaryDF.to_csv(path_or_buf=precisionF1Path, encoding='utf-8')

    try:
        if os.stat(sys.argv[5]).st_size > 0:
            # df_csv = pd.read_csv(sys.argv[5],encoding='utf-8',engine='python')
            # summaryDF = pd.concat([df_csv,summaryDF],axis=1,ignore_index=True)
            with open(sys.argv[5], 'a') as f:
                # Need to empty file contents now
                summaryDF.to_csv(path_or_buf=f, encoding='utf-8', mode='a', header=False)
                f.close()
        else:
            print "empty file"
            with open(sys.argv[5], 'w+') as f:
                summaryDF.to_csv(path_or_buf=f, encoding='utf-8')
                f.close()
    except OSError:
        print "No file"
        with open(sys.argv[5], 'w+') as f:
            summaryDF.to_csv(path_or_buf=f, encoding='utf-8')
            f.close()