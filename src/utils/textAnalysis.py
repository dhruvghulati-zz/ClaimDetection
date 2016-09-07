'''
This is a pipeline to produce the analyses we use in the thesis.

- The number of HTML files, the number of unique patterns extracted, the unique patterns, the number of location-value pairs before and after filtering.
- The number of training sentences extracted, from how many overall sentences as a ratio. The number of sentences we take as training.
- The statistics for the test sets.
- Analysis of the vocabulary after vectorization - the most common counts?
- Assess the property coverage in the test set and the training set.
- Comparison of the value distribution and their counts in training and test
- Comparison of the country distribution and their counts in training and test
- Comparison of the value distribution and their counts in training and KB
- Comparison of the country distribution and their counts in training and KB
- Text analysis - the % of the vocabulary covered by certain words and bigrams in training set.

/Users/dhruv/Documents/university/ClaimDetection/data/freebaseTriples.json
/Users/dhruv/Documents/university/ClaimDetection/data/output/sentenceRegionValue.json
/Users/dhruv/Documents/university/ClaimDetection/data/output/sentenceSlotsFiltered.json
/Users/dhruv/Documents/university/ClaimDetection/data/output/predictedPropertiesZero.json
/Users/dhruv/Documents/university/ClaimDetection/data/output/fullTestLabels.json
/Users/dhruv/Documents/university/ClaimDetection/data/output/cleanFullLabels.json
/Users/dhruv/Documents/university/ClaimDetection/data/sentenceMatrixFilteredZero.json
/Users/dhruv/Documents/university/ClaimDetection/data/output/uniqueSentenceLabels.json
/Users/dhruv/Documents/university/ClaimDetection/data/output/mainMatrixTrain.json
/Users/dhruv/Documents/university/ClaimDetection/data/output/mainMatrixTrain.json_sentences.json
/Users/dhruv/Documents/university/ClaimDetection/data/output/zero/data_analysis/

'''

import re
from nltk.corpus import stopwords  # Import the stop word list
import json
import sys
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
import string
import nltk
import numpy as np
from scipy import stats
import os, os.path
from numpy import histogram
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,ion,show

def numberFiles(directory):
    length = len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])
    print "Length is ",length
    return length

# normalize all words in a text by stripping out punctuation and whitespace
def text_normalizer(inputSentences):

    # empty list that will soon hold entire text as one list of words
    single_list = []

    # iterate through each line in the file
    for i, sentence in enumerate(inputSentences):

        line = sentence['parsedSentence']

        # punctuation-strip exception for "--" (em dash)
        if line.__contains__("--"):
            line = line.replace("--", " ")

        # replace hyphens with spaces
        if line.__contains__("-"):
            line = line.replace("-", " ")

        # strip all remaining punctuation
        for char in line:
            if char in string.punctuation:
                line = line.replace(char, "")


        line = line.lower()

        words = line.split()

        if words != []:
            single_list += words

    # return a single normalized list that contains words in the text
    return single_list

# takes a file and returns the number of words in it.
def wordcounter(f):
    return len(text_normalizer(f))


# prints the top 20 most frequent words in a file,
# followed by the no. of times each one appears
def top_twenty(f):
    # generate a histogram dictionary from the file
    d = word_histogram(f)

    # convert dictionary into a sorted (ascending) list
    # of tuples, where k = frequency and v = word
    lst_tup = sorted([(v, k) for k, v in d.iteritems()])

    i = -1

    # print last 20 (i.e. most frequent) items, formatted
    for num in range(20):
        print lst_tup[i][1] + ": " + str(lst_tup[i][0])
        i -= 1

def word_frequencies(f,threshold):
    # generate a histogram dictionary from the file
    d = word_histogram(f)

    count = float(len(text_normalizer(f)))

    # convert dictionary into a sorted (ascending) list
    # of tuples, where k = frequency and v = word
    lst_tup = sorted([(v/count, k) for k, v in d.iteritems()])

    i = -1
    cumsum = 0
    topwords = 0
    # print last 20 (i.e. most frequent) items, formatted
    while cumsum<float(threshold):
        cumsum += lst_tup[i][0]
        topwords = i/-1
        i -= 1

    return topwords, topwords/count

# takes a file, prints the no. of different words in
# the file, and returns a dict showing word frequency
def word_histogram(f):
    text = text_normalizer(f)
    d = {}
    for word in text:
        if word not in d:
            d[word] = 1
        else:
            d[word] += 1
    return d

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stemmer=PorterStemmer()
    stems = stem_tokens(tokens, stemmer)
    return stems

def sentence_to_words(sentence, remove_stopwords=False):
    letters_only = re.sub('[^a-zA-Z| LOCATION_SLOT | NUMBER_SLOT]', " ", sentence)
    # [x.strip() for x in re.findall('\s*(\w+|\W+)', line)]
    words = letters_only.lower().split()

    # In Python, searching a set is much faster than searching a list, so convert the stop words to a set
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return (words)

# This is about loading any file with property: region:value format


def training_features(inputSentences):
    global vectorizer
    # [:15000]
    for i, sentence in enumerate(inputSentences):
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
    print "Bag of words complete\n"

    return train_data_features


def test_features(testSentences):
    global vectorizer

    for sentence in testSentences:
        if sentence['parsedSentence'] != {} and sentence['mape_label'] != {}:
            test_wordlist.append(" ".join(sentence_to_words(sentence['parsedSentence'], True)))
            test_property_labels.append(sentence['property'])

    # print "These are the clean words in the test sentences: ", clean_test_sentences
    # print "These are the mape labels in the test sentences: ", binary_test_labels
    test_data_features = vectorizer.transform(clean_test_sentences)
    test_data_features = test_data_features.toarray()

    return test_data_features

if __name__ == "__main__":



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

    '''
    Now vectorize the features and create labels
    '''

    vectorizer = CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None, token_pattern="[\S]+",max_features=5000)

    print "Getting all the words in the training sentences...\n"


    #
    # train_data_features = training_features(pattern2regions)
    #
    # word_list = text_normalizer(pattern2regions)
    #
    # word_count = wordcounter(pattern2regions)
    #
    # print "Words per sentence is on average",word_count/len(pattern2regions),"\n"
    #
    # word_hist = word_histogram(pattern2regions)
    #
    # # print "Word histogram is ",word_hist
    #
    # top_twenty(pattern2regions)
    #
    # word_frequency, word_percent = word_frequencies(pattern2regions,0.95)
    #
    # print "Number of words accounting for threshold is ",word_frequency
    #
    # print 'Total % of vocab accounting for threshold is ', word_percent
    #
    # # stems_list = []
    # #
    # # for i, sentence in enumerate(pattern2regions):
    # #     stems = tokenize(sentence['parsedSentence'])
    # #     print "Stems are ",stems
    # #     stems_list.append(stems)
    #
    #
    # print "There are ",word_count,"words in the bag of words training sentences"
    #
    # print len(train_data_features), "sets of training features"
    #
    # trainingLabels = len(train_data_features)
    # positiveOpenTrainingLabels = len(train_property_labels_threshold) - train_property_labels_threshold.count(
    #     "no_region")
    # negativeOpenTrainingLabels = train_property_labels_threshold.count("no_region")
    # positiveClosedTrainingLabels = len(closed_train_property_labels_threshold) - closed_train_property_labels_threshold.count(
    #     "no_region")
    # negativeClosedTrainingLabels = closed_train_property_labels_threshold.count(
    #     "no_region")
    #
    # # Create an empty list and append the clean reviews one by one
    # clean_test_sentences = []
    #
    # print "Cleaning and parsing the test set ...\n"
    #
    # print "Get a bag of words for the test set, and convert to a numpy array\n"
    #
    # test_data_features = test_features(finalTestSentences)
    #
    # print len(test_data_features), "sets of test features"