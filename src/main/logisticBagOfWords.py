'''

This optimises hyperparameters both on probability and on the MAPE threshold and spits out results for:

- APE threshold versions of the predictors
- The probability prediction outputs for all predictors which can be reused.

python src/main/logisticBagOfWords.py data/output/predictedPropertiesZero.json data/output/fullTestLabels.json data/featuresKept.json data/output/zero/test data/output/zero/test/summaryEvaluation.csv

'''

import re
from nltk.corpus import stopwords  # Import the stop word list
import json
from nltk.util import ngrams
import numpy as np
from itertools import repeat
import pandas as pd
import copy
import sys
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
import os
from sklearn.pipeline import Pipeline

rng = np.random.RandomState(101)

# Balance my training data
# def balanced_subsample(x, y, subsample_size=1.0):
#     class_xs = []
#     min_elems = None
#
#     for yi in np.unique(y):
#         elems = x[(y == yi)]
#         class_xs.append((yi, elems))
#         if min_elems == None or elems.shape[0] < min_elems:
#             min_elems = elems.shape[0]
#
#     use_elems = min_elems
#     if subsample_size < 1:
#         use_elems = int(min_elems * subsample_size)
#
#     xs = []
#     ys = []
#
#     for ci, this_xs in class_xs:
#         if len(this_xs) > use_elems:
#             rng.shuffle(this_xs)
#
#         x_ = this_xs[:use_elems]
#         y_ = np.empty(use_elems)
#         y_.fill(ci)
#
#         xs.append(x_)
#         ys.append(y_)
#
#     xs = np.concatenate(xs)
#     ys = np.concatenate(ys)
#
#     return xs, ys


def sentence_to_words(sentence, remove_stopwords=False):
    letters_only = re.sub('[^a-zA-Z| LOCATION_SLOT | NUMBER_SLOT]', " ", sentence)
    words = letters_only.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return (words)

def find_ngrams(input_list, n):
  return zip(*[input_list[i:] for i in range(n)])


def training_features(inputSentences):
    global vectorizer
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

            train_wordbigram_list.append(words + " " + bigrams.decode("utf-8"))
            train_wordlist.append(words)

            wordgrams = find_ngrams(word_list,2)
            for i,grams in enumerate(wordgrams):
              wordgrams[i] = '+'.join(grams)
            wordgrams= (" ").join(wordgrams)
            train_gramlist.append(wordgrams)

            train_property_labels.append(sentence['predictedPropertyOpen'])
            train_property_labels_threshold.append(sentence['predictedPropertyOpenThreshold'])
            closed_train_property_labels.append(sentence['predictedPropertyClosed'])
            closed_train_property_labels_threshold.append(sentence['predictedPropertyClosedThreshold'])

def test_features(testSentences):
    global vectorizer

    for sentence in testSentences:
        if sentence['parsedSentence'] != {} and sentence['mape_label'] != {}:

            words = " ".join(sentence_to_words(sentence['parsedSentence'], True))
            word_list = sentence_to_words(sentence['parsedSentence'], True)
            wordgrams = find_ngrams(word_list,2)

            for i,grams in enumerate(wordgrams):
              wordgrams[i] = '+'.join(grams)
            wordgrams= (" ").join(wordgrams)
            test_gramlist.append(wordgrams)
            clean_test_sentences.append(words)
            test_property_labels.append(sentence['property'])
            bigrams = sentence['depPath']
            test_bigram_list.append(bigrams)
            test_wordbigram_list.append(words + " " + bigrams.decode("utf-8"))


# Find index of the true label for the sentence, and if that same index for that sentence is one, return the classifier class, else no region

if __name__ == "__main__":
    # training data
    # load the sentence file for training

    with open(sys.argv[1]) as trainingSentences:
        pattern2regions = json.loads(trainingSentences.read())

    print "We have ", len(pattern2regions), " training sentences."

    with open(sys.argv[2]) as testSentences:
        testSentences = json.loads(testSentences.read())

    # We load in the allowable features and also no_region
    with open(sys.argv[3]) as featuresKept:
        properties = json.loads(featuresKept.read())
    properties.append("no_region")

    finalTestSentences = []

    for sentence in testSentences:
        if sentence['parsedSentence'] != {} and sentence['mape_label'] != {} and sentence['mape'] != {} and sentence[
            'property'] != {} and sentence['property'] in properties:
            # print sentence['property']
            finalTestSentences.append(sentence)

    '''
    All the arrays needed to test.
    '''

    train_wordlist = []
    test_wordlist = []

    train_gramlist = []
    test_gramlist = []

    train_bigram_list = []
    test_bigram_list = []

    train_wordbigram_list = []
    test_wordbigram_list = []

    train_property_labels = []
    train_property_labels_threshold = []
    closed_train_property_labels = []
    closed_train_property_labels_threshold = []

    test_property_labels = []

    '''
    Define the pipeline
    '''
    # ('vect', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
    #                       ('tfidf', TfidfTransformer(use_idf=True,norm='l2',sublinear_tf=True)),
    text_clf = Pipeline([('vect', CountVectorizer(analyzer="word",token_pattern="[\S]+",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),('tfidf', TfidfTransformer(use_idf=True,norm='l2',sublinear_tf=True)),
                          ('clf',LogisticRegression(solver='newton-cg',class_weight='balanced', multi_class='multinomial',fit_intercept=True),
                          )])

    # This both sorts out the features and the training labels
    # train_data_features = training_features(pattern2regions)
    training_features(pattern2regions)

    clean_test_sentences = []

    print "Getting all the words in the training sentences...\n"
    #
    # test_data_features = test_features(finalTestSentences)
    test_features(finalTestSentences)
    #
    print len(clean_test_sentences), "sets of test features"
    #
    # print clean_test_sentences
    #
    # Load in the test data
    test = pd.DataFrame(finalTestSentences)


    # TODO This was an issue on command line - change to [2] if on command line. Should be 8 if testing.
    testSet = str(os.path.splitext(sys.argv[2])[0]).split("/")[2]

    test['testSet'] = np.array([testSet for x in range(len(test['property']))])

    threshold = test['threshold'][0]
    print "Theshold is ", threshold

    # This is outputted for the probability predictor
    testPath = os.path.join(sys.argv[4] + '/testData.csv')

    test.to_csv(path_or_buf=testPath, encoding='utf-8')

    # These are the ground truths
    y_multi_true = np.array(test['property'])
    y_true_claim = np.array(test['claim'])

    '''
    Get statistics on the training data in terms of counts
    '''
    # print len(train_data_features), "sets of training features"
    #
    trainingLabels = len(train_wordlist)
    positiveOpenTrainingLabels = len(train_property_labels_threshold) - train_property_labels_threshold.count(
        "no_region")
    negativeOpenTrainingLabels = train_property_labels_threshold.count("no_region")
    positiveClosedTrainingLabels = len(
        closed_train_property_labels_threshold) - closed_train_property_labels_threshold.count(
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

    # These are the random baselines
    unique_train_labels = set(train_property_labels)
    unique_train_labels_threshold = set(train_property_labels_threshold)
    closed_unique_train_labels = set(closed_train_property_labels)
    closed_unique_train_labels_threshold = set(closed_train_property_labels_threshold)

    openTrainingClasses = len(set(train_property_labels))
    openTrainingClassesThreshold = len(set(train_property_labels_threshold))
    closedTrainingClasses = len(set(closed_train_property_labels))
    closedTrainingClassesThreshold = len(set(closed_train_property_labels_threshold))
    #
    # print "There are ", len(set(train_property_labels)), "open training classes"
    # print "There are ", len(set(train_property_labels_threshold)), "open training classes w/threshold"
    # print "There are ", len(set(closed_train_property_labels)), "closed training properties"
    # print "There are ", len(set(closed_train_property_labels_threshold)), "closed training properties w/ threshold"

    # Fit the logistic classifiers to the training set, using the bag of words as features
    '''
    Fitting all classifiers
    '''


    print "Fitting the open multinomial BoW logistic regression model for probability models...\n"
    open_multi_logit_words = text_clf.fit(train_wordlist, train_property_labels)

    print "Fitting the open multinomial BoW logistic regression model w/ ",threshold," MAPE threshold...\n"
    open_multi_logit_threshold_words = copy.deepcopy(text_clf).fit(train_wordlist, train_property_labels_threshold)

    print "Fitting the closed multinomial BoW logistic regression model for probability models...\n"
    closed_multi_logit_words = copy.deepcopy(text_clf).fit(train_wordlist, closed_train_property_labels)

    print "Fitting the closed multinomial BoW logistic regression model w/ ",threshold," MAPE threshold...\n"
    closed_multi_logit_threshold_words = copy.deepcopy(text_clf).fit(train_wordlist, closed_train_property_labels_threshold)

    print "Fitting the open multinomial bigram logistic regression model for probability models...\n"
    open_multi_logit_bigrams = copy.deepcopy(text_clf).fit(train_gramlist, train_property_labels)

    print "Fitting the open multinomial bigram logistic regression model w/ ",threshold," MAPE threshold...\n"
    open_multi_logit_threshold_bigrams = copy.deepcopy(text_clf).fit(train_gramlist, train_property_labels_threshold)

    print "Fitting the closed multinomial bigram logistic regression model for probability models...\n"
    closed_multi_logit_bigrams = copy.deepcopy(text_clf).fit(train_gramlist, closed_train_property_labels)

    print "Fitting the closed multinomial bigram logistic regression model w/ ",threshold," MAPE threshold...\n"
    closed_multi_logit_threshold_bigrams = copy.deepcopy(text_clf).fit(train_gramlist, closed_train_property_labels_threshold)

    print "Fitting the open multinomial dependency bigram logistic regression model for probability models...\n"
    open_multi_logit_depgrams = copy.deepcopy(text_clf).fit(train_bigram_list, train_property_labels)

    print "Fitting the open multinomial dependency bigram logistic regression model w/ ",threshold," MAPE threshold...\n"
    open_multi_logit_threshold_depgrams = copy.deepcopy(text_clf).fit(train_bigram_list, train_property_labels_threshold)

    print "Fitting the closed multinomial dependency bigram logistic regression model for probability models...\n"
    closed_multi_logit_depgrams = copy.deepcopy(text_clf).fit(train_bigram_list, closed_train_property_labels)

    print "Fitting the closed multinomial dependency bigram logistic regression model w/ ",threshold," MAPE threshold...\n"
    closed_multi_logit_threshold_depgrams = copy.deepcopy(text_clf).fit(train_bigram_list, closed_train_property_labels_threshold)

    print "Fitting the open multinomial word+bigram logistic regression model for probability models...\n"
    open_multi_logit_wordgrams = copy.deepcopy(text_clf).fit(train_wordbigram_list, train_property_labels)

    print "Fitting the open multinomial word+bigram logistic regression model w/ ",threshold," MAPE threshold...\n"
    open_multi_logit_threshold_wordgrams = copy.deepcopy(text_clf).fit(train_wordbigram_list, train_property_labels_threshold)

    print "Fitting the closed multinomial word+bigram logistic regression model for probability models...\n"
    closed_multi_logit_wordgrams = copy.deepcopy(text_clf).fit(train_wordbigram_list, closed_train_property_labels)

    print "Fitting the closed multinomial word+bigram logistic regression model w/ ",threshold," MAPE threshold...\n"
    closed_multi_logit_threshold_wordgrams = copy.deepcopy(text_clf).fit(train_wordbigram_list, closed_train_property_labels_threshold)


    print "Saving the category mappings to files\n"

    multi_logit_categories = np.array(open_multi_logit_words.classes_)
    category_path = os.path.join(sys.argv[4] + '/open_categories.txt')
    np.savetxt(category_path, multi_logit_categories, fmt='%s')

    multi_logit_categories = np.array(open_multi_logit_threshold_words.classes_)
    category_path = os.path.join(sys.argv[4] + '/openthreshold_categories.txt')
    np.savetxt(category_path, multi_logit_categories, fmt='%s')

    multi_logit_categories = np.array(closed_multi_logit_words.classes_)
    category_path = os.path.join(sys.argv[4] + '/closed_categories.txt')
    np.savetxt(category_path, multi_logit_categories, fmt='%s')

    multi_logit_categories = np.array(closed_multi_logit_threshold_words.classes_)
    category_path = os.path.join(sys.argv[4] + '/closedthreshold_categories.txt')
    np.savetxt(category_path, multi_logit_categories, fmt='%s')
    #
    '''

    This is where the predictors start

    '''

    models = ['open_multi_logit_words', 'open_multi_logit_threshold_words', 'closed_multi_logit_words', 'closed_multi_logit_threshold_words',
              'open_multi_logit_bigrams', 'open_multi_logit_threshold_bigrams', 'closed_multi_logit_bigrams', 'closed_multi_logit_threshold_bigrams',
              'open_multi_logit_wordgrams', 'open_multi_logit_threshold_wordgrams', 'closed_multi_logit_wordgrams', 'closed_multi_logit_threshold_wordgrams',
              'open_multi_logit_depgrams', 'open_multi_logit_threshold_depgrams', 'closed_multi_logit_depgrams', 'closed_multi_logit_threshold_depgrams',
              'open_distant_sv_property_threshold', 'closed_distant_sv_property_threshold','andreas_threshold','categorical_random_threshold',
              'closed_categorical_random_threshold','test_data_mape_label', 'claim_label'
              ]


    categorical_random_threshold = rng.choice(list(unique_train_labels_threshold), len(finalTestSentences))
    closed_categorical_random_threshold = rng.choice(list(closed_unique_train_labels_threshold),len(finalTestSentences))

    model_data = {model:{} for model in models}

    # print model_data

    for model, dict in model_data.iteritems():
        dict['prediction'] = []
        dict['prob_prediction'] = []

        # Fill in the predictions
        if model=='open_multi_logit_words':
            dict['prediction'] = np.array(open_multi_logit_words.predict(clean_test_sentences)).tolist()
            dict['prob_prediction'] = np.array(open_multi_logit_words.predict_proba(clean_test_sentences)).tolist()
        if model=='closed_multi_logit_words':
            dict['prediction'] = np.array(closed_multi_logit_words.predict(clean_test_sentences)).tolist()
            dict['prob_prediction'] = np.array(closed_multi_logit_words.predict_proba(clean_test_sentences)).tolist()
        if model=='open_multi_logit_bigrams':
            dict['prediction'] = np.array(open_multi_logit_bigrams.predict(clean_test_sentences)).tolist()
            dict['prob_prediction'] = np.array(open_multi_logit_bigrams.predict_proba(clean_test_sentences)).tolist()
        if model=='closed_multi_logit_bigrams':
            dict['prediction'] = np.array(closed_multi_logit_bigrams.predict(clean_test_sentences)).tolist()
            dict['prob_prediction'] = np.array(closed_multi_logit_bigrams.predict_proba(clean_test_sentences)).tolist()
        if model=='open_multi_logit_depgrams':
            dict['prediction'] = np.array(open_multi_logit_depgrams.predict(clean_test_sentences)).tolist()
            dict['prob_prediction'] = np.array(open_multi_logit_depgrams.predict_proba(clean_test_sentences)).tolist()
        if model=='closed_multi_logit_depgrams':
            dict['prediction'] = np.array(closed_multi_logit_depgrams.predict(clean_test_sentences)).tolist()
            dict['prob_prediction'] = np.array(closed_multi_logit_depgrams.predict_proba(clean_test_sentences)).tolist()
        if model=='open_multi_logit_wordgrams':
            dict['prediction'] = np.array(open_multi_logit_wordgrams.predict(clean_test_sentences)).tolist()
            dict['prob_prediction'] = np.array(open_multi_logit_wordgrams.predict_proba(clean_test_sentences)).tolist()
        if model=='closed_multi_logit_wordgrams':
            dict['prediction'] = np.array(closed_multi_logit_wordgrams.predict(clean_test_sentences)).tolist()
            dict['prob_prediction'] = np.array(closed_multi_logit_wordgrams.predict_proba(clean_test_sentences)).tolist()

        if model=='open_multi_logit_threshold_words':
            dict['prediction'] = np.array(open_multi_logit_threshold_words.predict(clean_test_sentences)).tolist()
            dict['prob_prediction'] = np.array(open_multi_logit_threshold_words.predict_proba(clean_test_sentences)).tolist()
        if model=='closed_multi_logit_threshold_words':
            dict['prediction'] = np.array(closed_multi_logit_threshold_words.predict(clean_test_sentences)).tolist()
            dict['prob_prediction'] = np.array(closed_multi_logit_threshold_words.predict_proba(clean_test_sentences)).tolist()
        if model=='open_multi_logit_threshold_bigrams':
            dict['prediction'] = np.array(open_multi_logit_threshold_bigrams.predict(clean_test_sentences)).tolist()
            dict['prob_prediction'] = np.array(open_multi_logit_threshold_bigrams.predict_proba(clean_test_sentences)).tolist()
        if model=='closed_multi_logit_threshold_bigrams':
            dict['prediction'] = np.array(closed_multi_logit_threshold_bigrams.predict(clean_test_sentences)).tolist()
            dict['prob_prediction'] = np.array(closed_multi_logit_threshold_bigrams.predict_proba(clean_test_sentences)).tolist()
        if model=='open_multi_logit_threshold_depgrams':
            dict['prediction'] = np.array(open_multi_logit_threshold_depgrams.predict(clean_test_sentences)).tolist()
            dict['prob_prediction'] = np.array(open_multi_logit_threshold_depgrams.predict_proba(clean_test_sentences)).tolist()
        if model=='closed_multi_logit_threshold_depgrams':
            dict['prediction'] = np.array(closed_multi_logit_threshold_depgrams.predict(clean_test_sentences)).tolist()
            dict['prob_prediction'] = np.array(closed_multi_logit_threshold_depgrams.predict_proba(clean_test_sentences)).tolist()
        if model=='open_multi_logit_threshold_wordgrams':
            dict['prediction'] = np.array(open_multi_logit_threshold_wordgrams.predict(clean_test_sentences)).tolist()
            dict['prob_prediction'] = np.array(open_multi_logit_threshold_wordgrams.predict_proba(clean_test_sentences)).tolist()
        if model=='closed_multi_logit_threshold_wordgrams':
            dict['prediction'] = np.array(closed_multi_logit_threshold_wordgrams.predict(clean_test_sentences)).tolist()
            dict['prob_prediction'] = np.array(closed_multi_logit_threshold_wordgrams.predict_proba(clean_test_sentences)).tolist()
        if model=='open_distant_sv_property_threshold':
            dict['prediction'] = np.array(test['predictedPropertyOpenThreshold']).tolist()
        if model=='closed_distant_sv_property_threshold':
            dict['prediction'] = np.array(test['predictedPropertyClosedThreshold']).tolist()
        if model=='andreas_threshold':
            dict['prediction'] = np.array(test['categorical_mape_label']).tolist()
        if model=='categorical_random_threshold':
            dict['prediction'] = categorical_random_threshold.tolist()
        if model=='closed_categorical_random_threshold':
            dict['prediction'] = closed_categorical_random_threshold.tolist()
        if model=='test_data_mape_label':
            dict['binary_prediction'] = test['mape_label'].tolist()
        if model=='claim_label':
            dict['binary_prediction'] = y_true_claim.tolist()
    # print model_data

    # model_path = os.path.join(sys.argv[4] + '/models.json')
    #
    # # np.savetxt(model_path, model_data)
    #
    # with open(model_path, "wb") as out:
    #     json.dump(model_data, out,indent=4)

    categorical_data = {}

    for model, dict in model_data.iteritems():
        # dict['prob_path']=os.path.join(sys.argv[4] + model+'_prob_a.txt')
        # np.savetxt(dict['prob_path'], dict['prob_prediction'])
        if any(dict['prediction']):
            categorical_data[(model,'precision')]=precision_recall_fscore_support(y_multi_true,dict['prediction'],labels=list(set(y_multi_true)),average=None)[0]
            categorical_data[(model,'recall')]=precision_recall_fscore_support(y_multi_true,dict['prediction'],labels=list(set(y_multi_true)),average=None)[1]
            categorical_data[(model,'f1')]=precision_recall_fscore_support(y_multi_true,dict['prediction'],labels=list(set(y_multi_true)),average=None)[2]
            dict['binary_prediction']=[]
            for predict,true in zip(dict['prediction'],y_multi_true):
                if predict==true:
                    dict['binary_prediction'].append(1)
                else:
                    dict['binary_prediction'].append(0)

    categorical_data = pd.DataFrame(categorical_data,index=[item.split('/')[3] for item in list(set(y_multi_true))])

    categoricalPath = os.path.join(sys.argv[4] + testSet + '_' + str(threshold) + '_categoricalResults.csv')

    categorical_data.to_csv(path_or_buf=categoricalPath,encoding='utf-8')

    model_path = os.path.join(sys.argv[4] + '/models.json')

    # np.savetxt(model_path, model_data)

    with open(model_path, "wb") as out:
        json.dump(model_data, out,indent=4)


    output = {(outerKey, innerKey): values for outerKey, innerDict in model_data.iteritems() for innerKey, values in innerDict.iteritems() if innerKey=='prediction' or innerKey=='binary_prediction'}

    for (outerKey, innerKey),values in output.iteritems():
        if innerKey=='prediction' and values:
            print values
            split_values = np.array([str(item).split('/')[3] if item!="no_region" else item for item in values])
            output[(outerKey, innerKey)] = split_values
            # output[(outerKey, innerKey)]= [item.split('/')[3] for item in values]
        if innerKey == 'prediction' and not values:
            output[(outerKey, innerKey)] = np.array(["no_categorical_value" for x in range(len(finalTestSentences))])

    output = pd.DataFrame(output)

    # print "Output is ",output

    clean_test_sentences= np.array(clean_test_sentences)
    parsed_sentences =  np.array(test['parsedSentence'])
    test_bigram_list= np.array(test_bigram_list)
    test_gramlist= np.array(test_gramlist)
    test_wordbigram_list= np.array(test_wordbigram_list)
    threshold_array= np.array([threshold] * len(clean_test_sentences))

    data = {('global','features'): test_wordbigram_list,
            ('global','parsed_sentence'): parsed_sentences,
            ('global','depgrams'): test_bigram_list,
            ('global','bigrams'): test_gramlist,
            ('global','wordgrams'): test_wordbigram_list,
            ('global','threshold'):threshold_array,
            }

    # print "Data is",data

    DF = pd.DataFrame(data)

    # print "Data frame is",DF

    output = pd.concat([output,DF],axis=1)

    resultPath = os.path.join(sys.argv[4] + testSet + '_' + str(threshold) + '_regressionResult.csv')

    output.to_csv(path_or_buf=resultPath,encoding='utf-8')

    summaryDF = pd.DataFrame(columns=('precision', 'recall', 'f1', 'accuracy', 'evaluation set', 'threshold'))

    def evaluation(trueLabels, evalLabels, test_set, threshold):
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
                'probThreshold': ["no_probability_threshold"],
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

    for model,dict in model_data.iteritems():
        evaluation(y_true_claim, dict['binary_prediction'], testSet, threshold)

    # print summaryDF

    columns = list(model_data.keys())

    summaryDF.index = columns

    # print summaryDF
    #
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
