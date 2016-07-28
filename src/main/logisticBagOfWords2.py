'''

This optimises hyperparameters both on probability and on the MAPE threshold and spits out results for:



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


# Balance my training data
def balanced_subsample(x, y, subsample_size=1.0):
    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems * subsample_size)

    xs = []
    ys = []

    for ci, this_xs in class_xs:
        if len(this_xs) > use_elems:
            rng.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    return xs, ys


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
            train_property_labels_threshold.append(sentence['predictedPropertyOpenThreshold'])
            # Closed evaluation only include certain training sentences
            closed_train_property_labels.append(sentence['predictedPropertyClosed'])
            closed_train_property_labels_threshold.append(sentence['predictedPropertyClosedThreshold'])
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

    trainingLabels = len(train_data_features)
    positiveOpenTrainingLabels = len(train_property_labels_threshold) - train_property_labels_threshold.count(
        "no_region")
    negativeOpenTrainingLabels = train_property_labels_threshold.count("no_region")
    positiveClosedTrainingLabels = len(closed_train_property_labels_threshold) - closed_train_property_labels_threshold.count(
        "no_region")
    negativeClosedTrainingLabels = closed_train_property_labels_threshold.count(
        "no_region")


    print "There are ", len(train_property_labels_threshold) - train_property_labels_threshold.count(
        "no_region"), "positive open mape threshold labels"
    print "There are ", train_property_labels_threshold.count("no_region"), "negative open mape threshold labels"
    print "There are ", len(closed_train_property_labels_threshold) - closed_train_property_labels_threshold.count(
        "no_region"), "positive closed mape threshold labels"
    print "There are ", closed_train_property_labels_threshold.count(
        "no_region"), "negative closed mape threshold labels\n"

    # TODO This was an issue on command line - change to [2] if on command line
    testSet = str(os.path.splitext(sys.argv[2])[0]).split("/")[2]

    multi_logit = LogisticRegression(fit_intercept=True, class_weight='auto', multi_class='multinomial',
                                     solver='newton-cg')



    multi_logit_threshold = LogisticRegression(fit_intercept=True, class_weight='auto', multi_class='multinomial',
                                               solver='newton-cg')



    closed_multi_logit = LogisticRegression(fit_intercept=True, class_weight='auto', multi_class='multinomial',
                                            solver='newton-cg')


    closed_multi_logit_threshold = LogisticRegression(fit_intercept=True, class_weight='auto',
                                                      multi_class='multinomial',
                                                      solver='newton-cg')



    # Fit the logistic classifiers to the training set, using the bag of words as features

    openTrainingClasses = len(set(train_property_labels))
    openTrainingClassesThreshold = len(set(train_property_labels_threshold))
    closedTrainingClasses = len(set(closed_train_property_labels))
    closedTrainingClassesThreshold = len(set(closed_train_property_labels_threshold))


    print "There are ", len(set(train_property_labels)), "open training classes"
    print "There are ", len(set(train_property_labels_threshold)), "open training classes w/threshold"
    print "There are ", len(set(closed_train_property_labels)), "closed training properties"
    print "There are ", len(set(closed_train_property_labels_threshold)), "closed training properties w/ threshold"

    print "Fitting the open multinomial logistic regression model without MAPE threshold...\n"
    open_multi_logit = multi_logit.fit(train_data_features, train_property_labels)

    multi_logit_categories = np.array(open_multi_logit.classes_)
    category_path = os.path.join(sys.argv[4] +'/open_categories.txt')
    np.savetxt(category_path, multi_logit_categories,fmt='%s')


    print "Fitting the open multinomial logistic regression model w/ MAPE threshold...\n"
    open_threshold_multi_logit = multi_logit_threshold.fit(train_data_features, train_property_labels_threshold)

    multi_logit_threshold_categories = np.array(open_threshold_multi_logit.classes_)
    category_path = os.path.join(sys.argv[4] + '/openthreshold_categories.txt')
    np.savetxt(category_path, multi_logit_threshold_categories,fmt='%s')

    print "Fitting the closed multinomial logistic regression model without MAPE threshold...\n"
    closed_multi_logit = closed_multi_logit.fit(train_data_features, closed_train_property_labels)

    closed_multi_logit_categories = np.array(closed_multi_logit.classes_)
    category_path = os.path.join(sys.argv[4] +'/closed_categories.txt')
    np.savetxt(category_path, closed_multi_logit_categories,fmt='%s')

    print "Fitting the closed multinomial logistic regression model with MAPE threshold...\n"
    closed_threshold_multi_logit = closed_multi_logit_threshold.fit(train_data_features,
                                                                    closed_train_property_labels_threshold)

    closed_multi_logit_threshold_categories = np.array(closed_threshold_multi_logit.classes_)
    category_path = os.path.join(sys.argv[4] +'/closedthreshold_categories.txt')
    np.savetxt(category_path, closed_multi_logit_threshold_categories,fmt='%s')

    # Create an empty list and append the clean reviews one by one
    clean_test_sentences = []

    print "Cleaning and parsing the test set ...\n"

    print "Get a bag of words for the test set, and convert to a numpy array\n"

    test_data_features = test_features(finalTestSentences)

    print len(test_data_features), "sets of test features"

    print test_data_features

    # Load in the test data
    test = pd.DataFrame(finalTestSentences)

    test['testSet']=np.array([testSet for x in range(len(test['property']))])

    threshold = test['threshold'][0]
    print "Theshold is ", threshold

    # This is outputted for the probability predictor
    testPath = os.path.join(sys.argv[4] + '/testData.csv')

    test.to_csv(path_or_buf=testPath, encoding='utf-8')

    # These are the ground truths
    y_multi_true = np.array(test['property'])
    y_true_claim = np.array(test['claim'])

    '''

    This is where the predictors start

    '''

    print "Predicting open multinomial test labels w/ threshold...\n"
    y_multi_logit_result_open_threshold = np.array(open_threshold_multi_logit.predict(test_data_features))
    print "Predicting closed multinomial test labels w/ threshold...\n"
    y_multi_logit_result_closed_threshold = np.array(closed_threshold_multi_logit.predict(test_data_features))

    print "Predicting open multinomial probability argmax test labels...\n"

    prob_prediction = np.array(open_multi_logit.predict_proba(test_data_features))
    open_prob_path = os.path.join(sys.argv[4] +'/open_prob_a.txt')
    np.savetxt(open_prob_path, prob_prediction)

    print "Predicting open multinomial probability argmax test labels with threshold...\n"

    prob_prediction_threshold = np.array(open_threshold_multi_logit.predict_proba(test_data_features))
    open_prob_path_threshold = os.path.join(sys.argv[4] + '/open_prob_a_threshold.txt')
    np.savetxt(open_prob_path_threshold, prob_prediction_threshold)

    print "Predicting closed multinomial probability argmax test labels...\n"

    closed_prob_prediction = np.array(closed_multi_logit.predict_proba(test_data_features))
    closed_prob_path = os.path.join(sys.argv[4] + '/closed_prob_a.txt')
    np.savetxt(closed_prob_path, closed_prob_prediction)

    print "Predicting closed multinomial probability argmax test labels with threshold...\n"

    closed_prob_prediction_threshold = np.array(closed_threshold_multi_logit.predict_proba(test_data_features))
    closed_prob_path_threshold = os.path.join(sys.argv[4] + '/closed_prob_a_threshold.txt')
    np.savetxt(closed_prob_path_threshold, closed_prob_prediction_threshold)

    y_multi_logit_result_open_threshold_binary = []
    y_multi_logit_result_closed_threshold_binary = []

    # This is Andreas model for distant supervision
    y_andreas_mape = test['mape_label']

    y_distant_sv_property_openThreshold = test['predictedPropertyOpenThreshold']
    y_distant_sv_property_closedThreshold = test['predictedPropertyClosedThreshold']

    y_openThreshold_distant_sv_to_binary = []
    y_closedThreshold_distant_sv_to_binary = []

    positive_result = np.ones(len(finalTestSentences))
    y_pospred = positive_result

    catLabels = [
                 y_distant_sv_property_openThreshold,
                 y_distant_sv_property_closedThreshold,
                 y_multi_logit_result_open_threshold,
                 y_multi_logit_result_closed_threshold,
    ]

    binaryLabels = [
        y_openThreshold_distant_sv_to_binary,
        y_closedThreshold_distant_sv_to_binary,
        y_multi_logit_result_open_threshold_binary,
        y_multi_logit_result_closed_threshold_binary
    ]

    trueLabels = []
    trueLabels.extend(repeat(y_multi_true, len(binaryLabels)))


    # Convert the categorical predictions to binary based on if matching property
    def binaryConversion(trueLabel, evalLabel, binaryLabel):
        for true, eval in zip(trueLabel, evalLabel):
            # print true,eval, binary
            if eval == true:
                binaryLabel.append(1)
            else:
                binaryLabel.append(0)

                # print x,y,z


    for trueLabels, predictionLabels, emptyArray in zip(trueLabels, catLabels, binaryLabels):
        binaryConversion(trueLabels, predictionLabels, emptyArray)

    output = pd.DataFrame(data=dict(parsed_sentence=test['parsedSentence'],
                                    features=clean_test_sentences,

                                    open_property_prediction_withMAPEthreshold=y_multi_logit_result_open_threshold,
                                    open_property_prediction_withMAPEthreshold_toBinary=y_multi_logit_result_open_threshold_binary,
                                    closed_property_prediction_withMAPEthreshold=y_multi_logit_result_closed_threshold,
                                    closed_property_prediction_withMAPEthreshold_toBinary=y_multi_logit_result_closed_threshold_binary,

                                    distant_supervision_open_withMAPEThreshold=y_distant_sv_property_openThreshold,

                                    distant_supervision_closed_withMAPEThreshold=y_distant_sv_property_closedThreshold,
                                    distant_supervision_open_withMAPEThreshold_toBinary=y_openThreshold_distant_sv_to_binary,

                                    distant_supervision_closed_withMAPEThreshold_toBinary=y_closedThreshold_distant_sv_to_binary,

                                    test_data_mape_label=test['mape_label'],
                                    claim_label=y_true_claim,
                                    test_data_property_label=test['property'],
                                    andreas_prediction=y_pospred,
                                    threshold=np.full(len(y_true_claim), threshold)
                                    ))


    resultPath = os.path.join(sys.argv[4] + '/'+testSet + '_' + str(threshold) + '_' + '_regressionResult.csv')

    output.to_csv(path_or_buf=resultPath, encoding='utf-8', index=False, cols=[
        'parsed_sentence',
        'features',

        'open_property_prediction_withMAPEthreshold',
        'open_property_prediction_withMAPEthreshold_toBinary',
        'closed_property_prediction_withMAPEthreshold',
        'closed_property_prediction_withMAPEthreshold_toBinary',

        'distant_supervision_open_withMAPEThreshold',
        'distant_supervision_closed_withMAPEThreshold',
        'distant_supervision_open_withMAPEThreshold_toBinary',
        'distant_supervision_closed_withMAPEThreshold_toBinary',

        'test_data_mape_label',
        'claim_label',
        'andreas_property_label',
        'andreas_prediction',
        'threshold'
    ])

    # TODO - need to create a per property chart

    # Now we write our precision F1 etc to an Excel file
    summaryDF = pd.DataFrame(columns=('precision', 'recall', 'f1', 'accuracy', 'evaluation set', 'threshold'))


    def evaluation(trueLabels, evalLabels, test_set, threshold):
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
                'probThreshold': [""],
                'trainingLabels':[trainingLabels],
                'positiveOpenLabels':[positiveOpenTrainingLabels],
                'negativeOpenLabels':[negativeOpenTrainingLabels],
                'positiveClosedLabels':[positiveClosedTrainingLabels],
                'negativeClosedLabels':[negativeClosedTrainingLabels],
                'openTrainingClasses': [openTrainingClasses],
                'openTrainingClassesThreshold': [openTrainingClassesThreshold],
                'closedTrainingClasses':[closedTrainingClasses],
                'closedTrainingClassesThreshold': [closedTrainingClassesThreshold]
                }

        DF = pd.DataFrame(data)

        summaryDF = pd.concat([summaryDF, DF])


    results = [y_pospred,
               y_openThreshold_distant_sv_to_binary,
               y_closedThreshold_distant_sv_to_binary,
               y_multi_logit_result_open_threshold_binary,
               y_multi_logit_result_closed_threshold_binary

               ]
    #

    for result in results:
        evaluation(y_true_claim, result, testSet, threshold)

    columns = list([
                    'Previous_Model',
                    'Open_Property_MAPE_Threshold_Distant_Supervision_Model',
                    'Closed_Property_MAPE_Threshold_Distant_Supervision_Model',
                    'Open_Property_Bag_of_Words_Multinomial_Logistic_Regression_w_Binary_Evaluation_&_MAPE_Threshold',
                    'Closed_Property_Bag_of_Words_Multinomial_Logistic_Regression_w_Binary_Evaluation_&_MAPE_Threshold'
                    ])

    # summaryDF.set_index(['A','B'])
    summaryDF.index = columns

    print summaryDF
    #
    try:
        if os.stat(sys.argv[5]).st_size > 0:
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