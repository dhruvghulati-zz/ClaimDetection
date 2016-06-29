import re
from nltk.corpus import stopwords # Import the stop word list
import json
import numpy as np
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
import itertools
import pandas as pd
from sklearn import preprocessing
import os


rng = np.random.RandomState(101)
le = preprocessing.LabelEncoder()



# Tutorial from https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words
# python src/main/trainingFeatures.py data/output/predictedProperties.json data/output/hyperTestLabels.json data/regressionResult.json

'''TODO - get the features in a proper format to be able to do a logistic regression.
    Use class-weight = balanced
    http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    Cross validation
    Check size of training parameters
    Check how to do multinomial classifier
    To do - precision, recall, F1 for each region
'''

def sentence_to_words(sentence,remove_stopwords=False):
    # 2. Remove non-letters, and ensure location and number slots not split
    letters_only = re.sub("[^a-zA-Z| LOCATION_SLOT | NUMBER_SLOT]", " ", sentence)
    #
    # print letters_only
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()

    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    if remove_stopwords:
            stops = set(stopwords.words("english"))
            words = [w for w in words if not w in stops]
    return(words)

def training_features(inputSentences):

    global vectorizer

    # [:15000]
    for sentence in inputSentences:
    # Account for nones
        if sentence and sentence['predictedRegion'] in properties:
            train_wordlist.append(" ".join(sentence_to_words(sentence['parsedSentence'], True)))
            if sentence['predictedRegion']!="no_region":
                train_labels.append(1)
                train_property_labels.append(sentence['predictedRegion'])
            else:
                train_labels.append(0)
                train_property_labels.append("no_region")

    # print "These are the clean words in the training sentences: ", train_wordlist
    # print "These are the labels in the training sentences: ", train_labels

    print "Creating the bag of words...\n"

        # fit_transform() does two functions: First, it fits the model
        # and learns the vocabulary; second, it transforms our training data
        # into feature vectors. The input to fit_transform should be a list of
        # strings.
    train_data_features = vectorizer.fit_transform(train_wordlist)
        # Numpy arrays are easy to work with, so convert the result to an
        # array
    # '''TODO apparently I should keep this as a sparse matrix - see Pydata chat

    # '''
    train_data_features = train_data_features.toarray()

    return train_data_features


def test_features(testSentences):

    global vectorizer

    for sentence in testSentences:
        if sentence['parsedSentence']!={} and sentence['mape_label']!={}:
            clean_test_sentences.append(" ".join(sentence_to_words(sentence['parsedSentence'], True)))
            # print clean_test_sentences
            test_labels.append(sentence['mape_label'])
            test_property_labels.append(sentence['property'])

    print "These are the clean words in the test sentences: ", clean_test_sentences
    print "These are the labels in the test sentences: ", test_labels

    test_data_features = vectorizer.transform(clean_test_sentences)

    test_data_features = test_data_features.toarray()

    return test_data_features

# /Users/dhruv/Documents/university/ClaimDetection/data/output/predictedProperties.json
# /Users/dhruv/Documents/university/ClaimDetection/data/output/hyperTestLabels.json
# /Users/dhruv/Documents/university/ClaimDetection/data/regressionResult.json
if __name__ == "__main__":
    # training data
    # load the sentence file for training
    with open(sys.argv[1]) as trainingSentences:
        pattern2regions = json.loads(trainingSentences.read())

    with open(sys.argv[4]) as featuresKept:
        properties = json.loads(featuresKept.read())
    properties.append("no_region")

    print type(properties)
    for property in properties:
        print property

    print len(pattern2regions)," training sentences."

    with open(sys.argv[2]) as testSentences:
        testSentences = json.loads(testSentences.read())

    finalTestSentences = []

    for sentence in testSentences:
        '''
        TODO - need to make sure is empty before and clean, this is a messy step.
        '''
        if sentence['parsedSentence']!={} and sentence['mape_label']!={} and sentence['mape']!={} and sentence['property']!={} and sentence['property'] in properties:
            print sentence['property']
            # Todo - this seems to only be appending no_region
            finalTestSentences.append(sentence)

    vectorizer = CountVectorizer(analyzer = "word",   \
                         tokenizer = None,    \
                         preprocessor = None, \
                         stop_words = None,   \
                         max_features = 5000)

    # print pattern2regions

    print "Here are the final test sentences", finalTestSentences

    print "There are", len(finalTestSentences),"final test sentences"

    train_wordlist = []
    train_labels = []
    test_wordlist = []
    test_labels = []
    train_property_labels = []
    test_property_labels = []

    print "Getting all the words in the training sentences...\n"

    # # print "Training data features are: ", train_data_features
    #
    # This both sorts out the features and the training labels
    train_data_features = training_features(pattern2regions)

    print len(train_data_features),"sets of training features"

    # Initialize a Logistic Regression on the statistical region
    binary_logit = LogisticRegression(class_weight='auto')

    multi_logit = LogisticRegression(class_weight='auto', multi_class='multinomial', solver='lbfgs')

    # Fit the logistic classifiers to the training set, using the bag of words as
# features and the sentiment labels as the response variable
    #
    print "Fitting the binary logistic regression model\n"
    # This may take a few minutes to run
    # This seems to be wrong, based on empty training labels

    print train_data_features
    print "Training labels are ",train_labels

    binary_logit = binary_logit.fit(train_data_features, train_labels)

    print "These are the training labels\n"

    print train_property_labels

    print "Fitting the multinomial logistic regression model\n"

    le_train = le.fit(train_property_labels)
    train_classes = le_train.classes_

    print "These are our training classes\n",train_classes

    train_property_labels = le.transform(train_property_labels)

    multi_logit = multi_logit.fit(train_data_features, train_property_labels)

    # Create an empty list and append the clean reviews one by one
    clean_test_sentences = []

    print "Cleaning and parsing the test set ...\n"

    print "Get a bag of words for the test set, and convert to a numpy array\n"

    test_data_features = test_features(finalTestSentences)

    print len(test_data_features),"sets of test features"

    print test_data_features

    # Use the logistic regression to make predictions
    print "Predicting binary test labels...\n"

    binary_logit_result = binary_logit.predict(test_data_features)

    print "Predicting multinomial test labels...\n"

    multi_logit_result = multi_logit.predict(test_data_features)

    print "These are the property predictions\n"

    print multi_logit_result

    # These are the baselines

    random_result = rng.randint(2, size=len(finalTestSentences))

    positive_result = np.ones(len(finalTestSentences))

    negative_result = np.zeros(len(finalTestSentences))

    test = pd.DataFrame(finalTestSentences)

    # print test
    #
    # print test['parsedSentence']

    # Copy the results to a pandas dataframe with an "id" column and
    # a "sentiment" column
    # output = pd.DataFrame(data=dict(parsed_sentence=test['parsedSentence'], property=test['property'],
    #                                 predicted_label=binary_logit_result, random_label=random_result,
    #                                 actual_label=test['mape_label'],actual_property_label=test['property']))

    y_true = test['mape_label']
    y_true_claim = test['claim']
    print test['property']

    le_train = le.fit(train_property_labels)
    train_classes = le_train.classes_
    print "These are our train classes\n",train_classes
    # print len(set(test['property']))
    y_multi_true = le.transform(test['property'])

    print "These are the true classes", y_multi_true

    y_logpred = binary_logit_resul
    y_multilogpred = multi_logit_result
    y_randpred = random_result
    y_pospred = positive_result
    y_negpred = negative_result

    print "Precision, recall and F1 and support for binary logistic regression are ", precision_recall_fscore_support(y_true, y_logpred, pos_label=None,average='macro')

    # http://scikit-learn.org/stable/modules/model_evaluation.html#multiclass-and-multilabel-classification

    print "Precision, recall and F1 and support for multinomial logistic regression are ", precision_recall_fscore_support(y_multi_true, y_multilogpred, pos_label=None,average='macro')

    print "Precision, recall and F1 and support for random naive baseline are ", precision_recall_fscore_support(y_true_claim, y_randpred, pos_label=None,average='macro')

    print "Precision, recall and F1 and support for positive naive baseline are ", precision_recall_fscore_support(y_true_claim, y_pospred, pos_label=None,average='macro')

    print "Precision, recall and F1 and support for negative naive baseline are ", precision_recall_fscore_support(y_true_claim, y_negpred, pos_label=None,average='macro')


    # with open(sys.argv[3], "wb") as out:
    #     json.dump(output, out,indent=4)

    # Use pandas to write the comma-separated output file
    # output.to_csv(os.path.join(os.path.dirname(__file__), 'Bag_of_Words_model.csv'), index=False, escapechars=",",quoting=3, encoding="utf-8")
    # print "Wrote results to Bag_of_Words_model.csv"


