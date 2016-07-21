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
            # TODO - this not needed now
            if sentence['predictedPropertyClosedThreshold'] != "no_region":
                binary_closed_train_labels.append(1)
            else:
                binary_closed_train_labels.append(0)
            if sentence['predictedPropertyOpenThreshold'] != "no_region":
                binary_train_labels.append(1)
            else:
                binary_train_labels.append(0)
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


def probabilityThreshold(classifier, numberProperties, trainingFeatures, trainingLabels, testFeatures, testCatLabels):
    print "Number of properties is", numberProperties
    probThreshold = 1 / float(numberProperties)
    print "Threshold is", probThreshold

    # print testCatLabels

    fittedClassifier = classifier.fit(trainingFeatures, trainingLabels)
    prediction = np.array(fittedClassifier.predict_proba(testFeatures))

    categories = np.array(classifier.classes_)
    print "Categories are", categories
    print type(categories)

    # print type(prediction)
    # print "Probabilities are", prediction

    catIndex = []

    for i, label in enumerate(testCatLabels):
        index = [i for i, s in enumerate(categories) if s == label]
        catIndex.append(index)

    catIndex = np.array(catIndex).ravel()
    # print "Categorical indices are", catIndex
    # print type(catIndex)

    # Binarise the probabilities
    for i, sentence in enumerate(prediction):
        for j, prob in enumerate(sentence):
            if prob > probThreshold:
                prediction[i][j] = 1
            else:
                prediction[i][j] = 0

    # print "Binary labels are", prediction
    print "Number of 0s are ", prediction.size - np.count_nonzero(prediction)

    predictedBinaryValues = prediction[np.arange(len(catIndex)), catIndex]
    predictedCats = categories[catIndex]
    catResult = np.where(predictedBinaryValues, predictedCats, "no_region")

    # print catResult

    return catResult, predictedBinaryValues


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
    binary_train_labels = []
    binary_closed_train_labels = []
    test_wordlist = []
    binary_test_labels = []
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

    print "There are ", len(train_property_labels_threshold) - train_property_labels_threshold.count(
        "no_region"), "positive open mape threshold labels"
    print "There are ", train_property_labels_threshold.count("no_region"), "negative open mape threshold labels"
    print "There are ", len(closed_train_property_labels_threshold) - closed_train_property_labels_threshold.count(
        "no_region"), "positive closed mape threshold labels"
    print "There are ", closed_train_property_labels_threshold.count(
        "no_region"), "negative closed mape threshold labels\n"

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
    print "There are ", len(set(train_property_labels)), "open training classes"
    print "There are ", len(set(train_property_labels_threshold)), "open training classes w/threshold"
    print "There are ", len(set(closed_train_property_labels)), "closed training properties"
    print "There are ", len(set(closed_train_property_labels_threshold)), "closed training properties w/ threshold"

    print "Fitting the open multinomial logistic regression model without MAPE threshold...\n"
    open_multi_logit = multi_logit.fit(train_data_features, train_property_labels)
    print "Fitting the open multinomial logistic regression model w/ MAPE threshold...\n"
    open_threshold_multi_logit = multi_logit_threshold.fit(train_data_features, train_property_labels_threshold)
    print "Fitting the closed multinomial logistic regression model without MAPE threshold...\n"
    closed_multi_logit = closed_multi_logit.fit(train_data_features, closed_train_property_labels)
    print "Fitting the closed multinomial logistic regression model with MAPE threshold...\n"
    closed_threshold_multi_logit = closed_multi_logit_threshold.fit(train_data_features,
                                                                    closed_train_property_labels_threshold)

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
    print "Predicting open multinomial test labels w/ threshold...\n"
    y_multi_logit_result_open_threshold = np.array(open_threshold_multi_logit.predict(test_data_features))
    print "Predicting closed multinomial test labels w/ threshold...\n"
    y_multi_logit_result_closed = np.array(closed_multi_logit.predict(test_data_features))
    print "Predicting open multinomial test labels without threshold...\n"
    y_multi_logit_result_closed_threshold = np.array(closed_threshold_multi_logit.predict(test_data_features))

    # Load in the test data
    test = pd.DataFrame(finalTestSentences)
    threshold = test['threshold'][0]

    # These are the ground truths
    y_multi_true = np.array(test['property'])
    y_true_claim = np.array(test['claim'])

    print "Predicting open multinomial test labels with/without MAPE threshold using probability based predictor...\n"
    y_multi_logit_result_open_prob, y_multi_logit_result_open_prob_binary = probabilityThreshold(multi_logit, len(
        set(train_property_labels)), train_data_features, train_property_labels, test_data_features, y_multi_true)

    print "Predicting open multinomial test labels with/without MAPE threshold using probability based predictor...\n"
    y_multi_logit_result_closed_prob, y_multi_logit_result_closed_prob_binary = probabilityThreshold(closed_multi_logit,
                                                                                                     len(
                                                                                                         set(
                                                                                                             closed_train_property_labels)),
                                                                                                     train_data_features,
                                                                                                     closed_train_property_labels,
                                                                                                     test_data_features,
                                                                                                     y_multi_true)

    # TODO - this is not needed now
    # These are our predictions
    # y_logpred = binary_logit_result
    # y_multilogpred_binary = binary_multi_logit_result

    y_multi_logit_result_open_binary = []
    y_multi_logit_result_open_threshold_binary = []
    y_multi_logit_result_closed_binary = []
    y_multi_logit_result_closed_threshold_binary = []

    # This is Andreas model for distant supervision
    y_andreas_mape = test['mape_label']

    # TODO - we shouldn't have to say if 1 or 0 before'
    # This is the test labels for distant supervision
    # y_distant_sv_property_openMAPE = test['predictedOpenMAPELabel']
    # y_distant_sv_property_openThresholdMAPE = test['predictedOpenThresholdMAPELabel']
    # y_distant_sv_property_closedMAPE = test['predictedClosedMAPELabel']
    # y_distant_sv_property_closedThresholdMAPE = test['predictedClosedThresholdMAPELabel']

    y_distant_sv_property_open = test['predictedPropertyOpen']
    y_distant_sv_property_openThreshold = test['predictedPropertyOpenThreshold']
    y_distant_sv_property_closed = test['predictedPropertyClosed']
    y_distant_sv_property_closedThreshold = test['predictedPropertyClosedThreshold']

    y_open_distant_sv_to_binary = []
    y_closed_distant_sv_to_binary = []
    y_openThreshold_distant_sv_to_binary = []
    y_closedThreshold_distant_sv_to_binary = []

    # These are the random baselines
    unique_train_labels = set(train_property_labels)
    unique_train_labels_threshold = set(train_property_labels_threshold)
    closed_unique_train_labels = set(closed_train_property_labels)
    closed_unique_train_labels_threshold = set(closed_train_property_labels_threshold)

    # print unique_train_labels
    # Categorical random baseline
    categorical_random = rng.choice(list(unique_train_labels), len(finalTestSentences))
    categorical_random_threshold = rng.choice(list(unique_train_labels_threshold), len(finalTestSentences))
    closed_categorical_random = rng.choice(list(closed_unique_train_labels), len(finalTestSentences))
    closed_categorical_random_threshold = rng.choice(list(closed_unique_train_labels_threshold),
                                                     len(finalTestSentences))
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

    catLabels = [y_distant_sv_property_open, y_distant_sv_property_closed, y_distant_sv_property_openThreshold,
                 y_distant_sv_property_closedThreshold,
                 y_multi_logit_result_open,
                 y_multi_logit_result_open_threshold,
                 y_multi_logit_result_closed, y_multi_logit_result_closed_threshold,
                 categorical_random,
                 categorical_random_threshold,
                 closed_categorical_random,
                 closed_categorical_random_threshold]

    binaryLabels = [
        y_open_distant_sv_to_binary, y_closed_distant_sv_to_binary, y_openThreshold_distant_sv_to_binary,
        y_closedThreshold_distant_sv_to_binary,
        y_multi_logit_result_open_binary,
        y_multi_logit_result_open_threshold_binary,
        y_multi_logit_result_closed_binary, y_multi_logit_result_closed_threshold_binary,
        y_cat_random_to_binary, y_cat_random_to_binary_threshold, y_closed_random_to_binary,
        y_closedCat_random_to_binary_threshold]

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

                                    open_property_prediction=y_multi_logit_result_open,
                                    open_property_prediction_withMAPEthreshold=y_multi_logit_result_open_threshold,
                                    open_property_prediction_toBinary=y_multi_logit_result_open_binary,
                                    open_property_prediction_withMAPEthreshold_toBinary=y_multi_logit_result_open_threshold_binary,
                                    closed_property_prediction=y_multi_logit_result_closed,
                                    closed_property_prediction_withMAPEthreshold=y_multi_logit_result_closed_threshold,
                                    closed_property_prediction_toBinary=y_multi_logit_result_closed_binary,
                                    closed_property_prediction_withMAPEthreshold_toBinary=y_multi_logit_result_closed_threshold_binary,

                                    open_property_probability_prediction=y_multi_logit_result_open_prob,
                                    open_property_probability_prediction_toBinary=y_multi_logit_result_open_prob_binary,
                                    closed_property_probability_prediction=y_multi_logit_result_closed_prob,
                                    closed_property_probability_prediction_toBinary=y_multi_logit_result_closed_prob_binary,

                                    distant_supervision_open=y_distant_sv_property_open,
                                    distant_supervision_open_withMAPEThreshold=y_distant_sv_property_openThreshold,
                                    distant_supervision_closed=y_distant_sv_property_closed,
                                    distant_supervision_closed_withMAPEThreshold=y_distant_sv_property_closedThreshold,
                                    distant_supervision_open_toBinary=y_open_distant_sv_to_binary,
                                    distant_supervision_open_withMAPEThreshold_toBinary=y_openThreshold_distant_sv_to_binary,
                                    distant_supervision_closed_toBinary=y_closed_distant_sv_to_binary,
                                    distant_supervision_closed_withMAPEThreshold_toBinary=y_closedThreshold_distant_sv_to_binary,

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
                                    negative_baseline=y_negpred,
                                    threshold=np.full(len(y_true_claim), threshold)))

    # print str(os.path.splitext(sys.argv[2])[0]).split("/")
    # TODO This was an issue on command line - change to [2] if on command line
    testSet = str(os.path.splitext(sys.argv[2])[0]).split("/")[2]

    resultPath = os.path.join(sys.argv[4] + "test/" + testSet + '_' + str(threshold) + '_regressionResult.csv')

    output.to_csv(path_or_buf=resultPath, encoding='utf-8', index=False, cols=[
        'parsed_sentence',
        'features',
        'open_property_prediction',
        'open_property_prediction_withMAPEthreshold',
        'open_property_prediction_toBinary',
        'open_property_prediction_withMAPEthreshold_toBinary',
        'closed_property_prediction',
        'closed_property_prediction_withMAPEthreshold',
        'closed_property_prediction_toBinary',
        'closed_property_prediction_withMAPEthreshold_toBinary',

        'open_property_probability_prediction',
        'closed_property_probability_prediction',

        'distant_supervision_open',
        'distant_supervision_open_withMAPEThreshold',
        'distant_supervision_closed',
        'distant_supervision_closed_withMAPEThreshold',
        'distant_supervision_open_toBinary',
        'distant_supervision_open_withMAPEThreshold_toBinary',
        'distant_supervision_closed_toBinary',
        'distant_supervision_closed_withMAPEThreshold_toBinary',

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
        'threshold'
    ])

    # TODO - need to create a per property chart

    # Now we write our precision F1 etc to an Excel file
    summaryDF = pd.DataFrame(columns=('precision', 'recall', 'f1', 'accuracy', 'evaluation set', 'threshold'))


    def evaluation(trueLabels, evalLabels, test_set, threshold):
        global summaryDF

        precision = precision_score(trueLabels, evalLabels)
        recall = recall_score(trueLabels, evalLabels)
        f1 = f1_score(trueLabels, evalLabels)
        accuracy = accuracy_score(trueLabels, evalLabels)

        data = {'precision': [precision],
                'recall': [recall],
                'f1': [f1],
                'accuracy': [accuracy],
                'evaluation set': [test_set],
                'threshold': [threshold]
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
               y_openThreshold_distant_sv_to_binary,
               y_closed_distant_sv_to_binary,
               y_closedThreshold_distant_sv_to_binary,
               y_multi_logit_result_open_binary,
               y_multi_logit_result_open_threshold_binary,
               y_multi_logit_result_closed_binary,
               y_multi_logit_result_closed_threshold_binary,
               y_multi_logit_result_open_prob_binary,
               y_multi_logit_result_closed_prob_binary
               ]
    #

    for result in results:
        evaluation(y_true_claim, result, testSet, threshold)

    columns = list(['Binary_Random_Baseline',
                    'Open_Categorical_Random_Baseline',
                    'Open_Categorical_Random Baseline_w_Threshold',
                    'Closed_Categorical_Random Baseline',
                    'Closed_Categorical_Random Baseline_w_Threshold',
                    'Previous_Model',
                    'Negative_Naive_Baseline',
                    'Open_Property_Distant_Supervision_Model',
                    'Open_Property_MAPE_Threshold_Distant_Supervision_Model',
                    'Closed_Property_Distant_Supervision_Model',
                    'Closed_Property_MAPE_Threshold_Distant_Supervision_Model',
                    'Open_Property_Bag_of_Words_Multinomial_Logistic_Regression_w_Binary_Evaluation',
                    'Open_Property_Bag_of_Words_Multinomial_Logistic_Regression_w_Binary_Evaluation_&_MAPE_Threshold',
                    'Closed_Property_Bag_of_Words_Multinomial_Logistic_Regression_w_Binary_Evaluation',
                    'Closed_Property_Bag_of_Words_Multinomial_Logistic_Regression_w_Binary_Evaluation_&_MAPE_Threshold',
                    'Open_Property_Probability_Prediction',
                    'Closed_Property_Probability_Prediction'
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