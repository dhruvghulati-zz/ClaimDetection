'''

This file predicts only once for models that do not change with hyperparameters:

Closed_Property_Bag_of_Words_Multinomial_Logistic_Regression_w_Binary_Evaluation
Open_Property_Bag_of_Words_Multinomial_Logistic_Regression_w_Binary_Evaluation

Closed_Property_Bigrams_Multinomial_Logistic_Regression_w_Binary_Evaluation
Open_Property_Bigrams_Multinomial_Logistic_Regression_w_Binary_Evaluation

Closed_Property_Depgrams_Multinomial_Logistic_Regression_w_Binary_Evaluation
Open_Property_Depgrams_Multinomial_Logistic_Regression_w_Binary_Evaluation

Closed_Property_Wordgrams_Multinomial_Logistic_Regression_w_Binary_Evaluation
Open_Property_Wordgrams_Multinomial_Logistic_Regression_w_Binary_Evaluation

Open_Property_Distant_Supervision_Model
Closed_Property_Distant_Supervision_Model

Random Binary Baseline

Open_Categorical_Random_Baseline
Open_Categorical_Random Baseline_w_Threshold
Closed_Categorical_Random Baseline
Closed_Categorical_Random Baseline_w_Threshold

Negative_Naive_Baseline

Previous_Model

Parameters
--------------

data/output/predictedPropertiesZero.json
data/output/fullTestLabels.json
data/featuresKept.json
data/output/zero/test/
data/output/zero/test/summaryEvaluation.csv

python src/main/naiveBaselines.py data/output/predictedPropertiesZero.json data/output/fullTestLabels.json data/featuresKept.json data/output/zero/test/ data/output/zero/test/summaryEvaluation.csv


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
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
import os
import copy

from sklearn.pipeline import Pipeline

rng = np.random.RandomState(101)


def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])


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
    for i, sentence in enumerate(inputSentences):
        # Dont train if the sentence contains a random region we don't care about
        # and sentence['predictedRegion'] in properties
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

            wordgrams = find_ngrams(word_list, 2)
            for i, grams in enumerate(wordgrams):
                wordgrams[i] = '+'.join(grams)
            wordgrams = (" ").join(wordgrams)
            train_gramlist.append(wordgrams)

            train_property_labels.append(sentence['predictedPropertyOpen'])
            closed_train_property_labels.append(sentence['predictedPropertyClosed'])
            # print "These are the clean words in the training sentences: ", train_wordlist
            # print "These are the labels in the training sentences: ", train_labels
            # This is for vectorizing
            # print "Creating the bag of words...\n"
            # train_data_features = vectorizer.fit_transform(train_wordlist)
            # train_data_features = train_data_features.toarray()
            #
            # return train_data_features

def change(word):
    # print "Word is",word
    if word=="no_region":
        return word
    else:
        return word.split('/')[3]

def test_features(testSentences):
    global vectorizer

    for sentence in testSentences:
        words = " ".join(sentence_to_words(sentence['parsedSentence'], True))
        word_list = sentence_to_words(sentence['parsedSentence'], True)
        wordgrams = find_ngrams(word_list, 2)

        for i, grams in enumerate(wordgrams):
            wordgrams[i] = '+'.join(grams)
        wordgrams = (" ").join(wordgrams)
        test_gramlist.append(wordgrams)
        clean_test_sentences.append(words)
        test_property_labels.append(sentence['property'])

        bigrams = ""
        if "depPath" in sentence.keys():
            bigrams = sentence['depPath'].encode('utf-8')

        test_bigram_list.append(bigrams)
        test_wordbigram_list.append(words + " " + bigrams.decode("utf-8"))

            # print "These are the clean words in the test sentences: ", clean_test_sentences
            # print "These are the mape labels in the test sentences: ", binary_test_labels
            # test_data_features = vectorizer.transform(clean_test_sentences)
            # test_data_features = test_data_features.toarray()
            #
            # return test_data_features


# Find index of the true label for the sentence, and if that same index for that sentence is one, return the classifier class, else no region


def uniqify(seq, idfun=None):
   # order preserving
   if idfun is None:
       def idfun(x): return x
   seen = {}
   result = []
   for item in seq:
       marker = idfun(item)
       # in old Python versions:
       # if seen.has_key(marker)
       # but in new ones:
       if marker in seen: continue
       seen[marker] = 1
       result.append(item)
   return result

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

    train_wordlist = []
    test_wordlist = []

    train_gramlist = []
    test_gramlist = []

    train_bigram_list = []
    test_bigram_list = []

    train_wordbigram_list = []
    test_wordbigram_list = []

    train_property_labels = []
    closed_train_property_labels = []

    test_property_labels = []

    # vectorizer = CountVectorizer(analyzer="word", \
    #                              tokenizer=None, \
    #                              preprocessor=None, \
    #                              stop_words=None, \
    #                              max_features=5000)

    print "Getting all the words in the training sentences...\n"

    # This both sorts out the features and the training labels
    training_features(pattern2regions)

    # print str(os.path.splitext(sys.argv[2])[0]).split("/")
    # TODO This was an issue on command line - change to [2] if on command line, 8 if not
    testSet = str(os.path.splitext(sys.argv[2])[0]).split("/")[2]

    print "Test set is",testSet

    print len(train_wordlist), "sets of training features"

    # Fit the logistic classifiers to the training set, using the bag of words as features
    print "There are ", len(set(train_property_labels)), "open training classes"
    print "There are ", len(set(closed_train_property_labels)), "closed training properties"

    # Create an empty list and append the clean reviews one by one
    clean_test_sentences = []

    print "Get a bag of words for the test set, and convert to a numpy array\n"

    test_features(testSentences)

    print len(clean_test_sentences), "sets of test features"

    text_clf = Pipeline([('vect', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                         ('tfidf', TfidfTransformer(use_idf=True,norm='l2',sublinear_tf=True)),
                          ('clf',LogisticRegression(solver='newton-cg',class_weight='balanced', multi_class='multinomial',fit_intercept=True),
                          )])

    trainingLabels = len(train_wordlist)


    print "Fitting the open multinomial BoW logistic regression model...\n"
    open_multi_logit_words = text_clf.fit(train_wordlist, train_property_labels)

    print "Fitting the closed multinomial BoW logistic regression model...\n"
    closed_multi_logit_words = copy.deepcopy(text_clf).fit(train_wordlist, closed_train_property_labels)

    print "Fitting the open multinomial bigram logistic regression model...\n"
    open_multi_logit_bigrams = copy.deepcopy(text_clf).fit(train_gramlist, train_property_labels)

    print "Fitting the closed multinomial bigram logistic regression model...\n"
    closed_multi_logit_bigrams = copy.deepcopy(text_clf).fit(train_gramlist, closed_train_property_labels)

    print "Fitting the open multinomial dependency bigram logistic regression model...\n"
    open_multi_logit_depgrams = copy.deepcopy(text_clf).fit(train_bigram_list, train_property_labels)

    print "Fitting the closed multinomial dependency bigram logistic regression model...\n"
    closed_multi_logit_depgrams = copy.deepcopy(text_clf).fit(train_bigram_list, closed_train_property_labels)

    print "Fitting the open multinomial word+bigram logistic regression model...\n"
    open_multi_logit_wordgrams = copy.deepcopy(text_clf).fit(train_wordbigram_list, train_property_labels)

    print "Fitting the closed multinomial word+bigram logistic regression model...\n"
    closed_multi_logit_wordgrams = copy.deepcopy(text_clf).fit(train_wordbigram_list, closed_train_property_labels)

    print "Saving the category mappings to files\n"

    multi_logit_categories = np.array(open_multi_logit_words.classes_)
    category_path = os.path.join(sys.argv[4] + 'open_categories.txt')
    np.savetxt(category_path, multi_logit_categories, fmt='%s')

    multi_logit_categories = np.array(closed_multi_logit_words.classes_)
    category_path = os.path.join(sys.argv[4] + 'closed_categories.txt')
    np.savetxt(category_path, multi_logit_categories, fmt='%s')

    # 'test_data_mape_label',TODO - this is not in the clean labels
    # TODO - distant supervision doesn't work for clean labels as many contain useless regions
    # 'distant_supervision_open','distant_supervision_closed',
    models = ['open_multi_logit_words', 'closed_multi_logit_words',
              'open_multi_logit_bigrams', 'closed_multi_logit_bigrams',
              'open_multi_logit_wordgrams', 'closed_multi_logit_wordgrams',
              'open_multi_logit_depgrams', 'closed_multi_logit_depgrams',
                'random_binary_label',
              'random_categorical_label', 'closed_random_categorical_label', 'claim_label',
              'andreas_property_label', 'andreas_prediction', 'negative_baseline',
              ]

    model_data = {model: {} for model in models}

    # Load in the test data
    test = pd.DataFrame(testSentences)

    test['testSet'] = np.array([testSet for x in range(len(test['property']))])

    # This is outputted for the probability predictor
    testPath = os.path.join(sys.argv[4] + 'testData.csv')

    test.to_csv(path_or_buf=testPath, encoding='utf-8')

    # These are the ground truths
    y_multi_true = np.array(test['property'])
    # print "True labels are",y_multi_true
    y_true_claim = np.array(test['claim'])
    unique_test_labels = uniqify(y_multi_true)

    # print "Here are the list of unique true labels",(unique_test_labels)

    # This is Andreas model for distant supervision
    # y_andreas_mape = test['mape_label']
    # TODO - in certain datasets this doesn't exist because it hasn't been applied
    y_distant_sv_property_open = test['predictedPropertyOpen']
    y_distant_sv_property_closed = test['predictedPropertyClosed']

    # These are the random baselines
    unique_train_labels = set(train_property_labels)
    unique_train_labels_threshold = unique_train_labels.union(['no_region'])
    closed_unique_train_labels = set(closed_train_property_labels)
    closed_unique_train_labels_threshold = closed_unique_train_labels.union(['no_region'])

    openTrainingClasses = len(set(train_property_labels))
    closedTrainingClasses = len(set(closed_train_property_labels))

    categorical_random = rng.choice(list(unique_train_labels), len(testSentences))
    categorical_random_threshold = rng.choice(list(unique_train_labels_threshold), len(testSentences))
    closed_categorical_random = rng.choice(list(closed_unique_train_labels), len(testSentences))
    closed_categorical_random_threshold = rng.choice(list(closed_unique_train_labels_threshold),
                                                     len(testSentences))

    # Random 0 and 1
    random_result = rng.randint(2, size=len(testSentences))
    positive_result = np.ones(len(testSentences))
    negative_result = np.zeros(len(testSentences))

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

        if model == 'distant_supervision_open':
            dict['prediction'] = y_distant_sv_property_open.tolist()
        if model == 'distant_supervision_closed':
            dict['prediction'] = y_distant_sv_property_closed.tolist()
        if model == 'random_binary_label':
            dict['binary_prediction'] = random_result.tolist()
        if model == 'random_categorical_label':
            dict['prediction'] = categorical_random.tolist()
        if model == 'closed_random_categorical_label':
            dict['prediction'] = closed_categorical_random.tolist()
        if model == 'test_data_mape_label':
            dict['binary_prediction'] = test['mape_label'].tolist()
        if model == 'claim_label':
            dict['binary_prediction'] = y_true_claim.tolist()
        if model == 'andreas_property_label':
            dict['prediction'] = test['property'].tolist()
        if model == 'andreas_prediction':
            dict['binary_prediction'] = positive_result.tolist()
        if model == 'negative_baseline':
            dict['binary_prediction'] = negative_result.tolist()

    # print model_data
    # This tells the probability predictor
    model_path = os.path.join(sys.argv[4] + 'models.json')

    with open(model_path, "wb") as out:
        json.dump(model_data, out)

    categorical_data = {}

    for model, dict in model_data.iteritems():
        if any(dict['prediction']):
            categorical_data[(model, 'precision')] = \
            precision_recall_fscore_support(y_multi_true, dict['prediction'], labels=list(set(y_multi_true)),
                                            average=None)[0]
            categorical_data[(model, 'recall')] = \
            precision_recall_fscore_support(y_multi_true, dict['prediction'], labels=list(set(y_multi_true)),
                                            average=None)[1]
            categorical_data[(model, 'f1')] = \
            precision_recall_fscore_support(y_multi_true, dict['prediction'], labels=list(set(y_multi_true)),
                                            average=None)[2]
            dict['binary_prediction'] = []
            for predict, true in zip(dict['prediction'], y_multi_true):
                # This needs to account for also if the prediction was "no_region"
                if predict == true:
                    if predict=="no_region":
                        # Predicting no claim
                        dict['binary_prediction'].append(0)
                    else:
                        dict['binary_prediction'].append(1)
                else:
                    dict['binary_prediction'].append(0)

    # This accounts for the fact that no region cannot be split
    categorical_data = pd.DataFrame(categorical_data, index=[change(item) for item in list(set(y_multi_true))])

    categoricalPath = os.path.join(sys.argv[4] + testSet + '_no_threshold_categoricalResults.csv')

    categorical_data.to_csv(path_or_buf=categoricalPath, encoding='utf-8')

    output = {(outerKey, innerKey): values for outerKey, innerDict in model_data.iteritems() for innerKey, values in
              innerDict.iteritems() if innerKey == 'prediction' or innerKey == 'binary_prediction'}

    for (outerKey, innerKey), values in output.iteritems():
        if innerKey == 'prediction' and values:
            # print outerKey,values
            split_values = np.array([change(item) for item in values])
            output[(outerKey, innerKey)] = split_values
        if innerKey == 'prediction' and not values:
            output[(outerKey, innerKey)] = np.array(["no_categorical_value" for x in range(len(testSentences))])

    output = pd.DataFrame(output)

    clean_test_sentences = np.array(clean_test_sentences)
    parsed_sentences = np.array(test['parsedSentence'])
    test_bigram_list = np.array(test_bigram_list)
    test_gramlist = np.array(test_gramlist)
    test_wordbigram_list = np.array(test_wordbigram_list)
    threshold_array = np.array(["no_threshold"] * len(clean_test_sentences))

    data = {('global', 'features'): test_wordbigram_list,
            ('global', 'parsed_sentence'): parsed_sentences,
            ('global', 'depgrams'): test_bigram_list,
            ('global', 'bigrams'): test_gramlist,
            ('global', 'wordgrams'): test_wordbigram_list,
            ('global', 'threshold'): threshold_array,
            }

    DF = pd.DataFrame(data)

    output = pd.concat([output, DF], axis=1)

    resultPath = os.path.join(sys.argv[4] + testSet + '_no_threshold'+'_regressionResult.csv')

    output.to_csv(path_or_buf=resultPath, encoding='utf-8')

    summaryDF = pd.DataFrame(columns=('precision', 'recall', 'f1', 'accuracy', 'evaluation set'))


    def evaluation(trueLabels, evalLabels, test_set):
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
                'probThreshold': ["no_probability_threshold"],
                'trainingLabels': [trainingLabels],
                'positiveOpenLabels': [""],
                'negativeOpenLabels': [""],
                'positiveClosedLabels': [""],
                'negativeClosedLabels': [""],
                'openTrainingClasses': [openTrainingClasses],
                'openTrainingClassesThreshold': [""],
                'closedTrainingClasses': [closedTrainingClasses],
                'closedTrainingClassesThreshold': [""]
                }

        DF = pd.DataFrame(data)

        summaryDF = pd.concat([summaryDF, DF])


    for model, dict in model_data.iteritems():
        evaluation(y_true_claim, dict['binary_prediction'], testSet)

    columns = list(model_data.keys())

    summaryDF.index = columns

    try:
        if os.stat(sys.argv[5]).st_size > 0:
            with open(sys.argv[5], 'a') as f:
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
