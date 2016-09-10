'''

This is the source of all the charts in my thesis.

/Users/dhruv/Documents/university/ClaimDetection/data/freebaseTriples.json
/Users/dhruv/Documents/university/ClaimDetection/data/output/predictedPropertiesZero.json
/Users/dhruv/Documents/university/ClaimDetection/data/output/fullTestLabels.json
/Users/dhruv/Documents/university/ClaimDetection/data/output/cleanFullLabels.json
/Users/dhruv/Documents/university/ClaimDetection/data/mainMatrixFiltered.json
/Users/dhruv/Documents/university/ClaimDetection/data/theMatrixExtend120TokenFiltered_2_2_0.1_0.5_fixed2.json
/Users/dhruv/Documents/university/ClaimDetection/figures/

'''

#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import os
import sys
from collections import Counter
from operator import itemgetter
import matplotlib.pyplot as plt
import nltk
import numpy as np
import json

from nltk import PorterStemmer
from nltk.corpus import stopwords
from astroML.plotting import hist
from scipy.stats import stats
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
from sklearn.pipeline import Pipeline
import re
from scipy.special import expit

# def numberFiles(directory):
#     length = len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])
#     print "Length is ",length
#     return length
#
# # normalize all words in a text by stripping out punctuation and whitespace
# def text_normalizer(inputSentences):
#
#     # empty list that will soon hold entire text as one list of words
#     single_list = []
#
#     # iterate through each line in the file
#     for i, sentence in enumerate(inputSentences):
#
#         line = sentence['parsedSentence']
#
#         # punctuation-strip exception for "--" (em dash)
#         if line.__contains__("--"):
#             line = line.replace("--", " ")
#
#         # replace hyphens with spaces
#         if line.__contains__("-"):
#             line = line.replace("-", " ")
#
#         # strip all remaining punctuation
#         for char in line:
#             if char in string.punctuation:
#                 line = line.replace(char, "")
#
#
#         line = line.lower()
#
#         words = line.split()
#
#         if words != []:
#             single_list += words
#
#     # return a single normalized list that contains words in the text
#     return single_list
#
# # takes a file and returns the number of words in it.
# def wordcounter(f):
#     return len(text_normalizer(f))
#
#
# # prints the top 20 most frequent words in a file,
# # followed by the no. of times each one appears
# def top_twenty(f):
#     # generate a histogram dictionary from the file
#     d = word_histogram(f)
#
#     # convert dictionary into a sorted (ascending) list
#     # of tuples, where k = frequency and v = word
#     lst_tup = sorted([(v, k) for k, v in d.iteritems()])
#
#     i = -1
#
#     # print last 20 (i.e. most frequent) items, formatted
#     for num in range(20):
#         print lst_tup[i][1] + ": " + str(lst_tup[i][0])
#         i -= 1
#
# def word_frequencies(f,threshold):
#     # generate a histogram dictionary from the file
#     d = word_histogram(f)
#
#     count = float(len(text_normalizer(f)))
#
#     # convert dictionary into a sorted (ascending) list
#     # of tuples, where k = frequency and v = word
#     lst_tup = sorted([(v/count, k) for k, v in d.iteritems()])
#
#     i = -1
#     cumsum = 0
#     topwords = 0
#     # print last 20 (i.e. most frequent) items, formatted
#     while cumsum<float(threshold):
#         cumsum += lst_tup[i][0]
#         topwords = i/-1
#         i -= 1
#
#     return topwords, topwords/count
#
# # takes a file, prints the no. of different words in
# # the file, and returns a dict showing word frequency
# def word_histogram(f):
#     text = text_normalizer(f)
#     d = {}
#     for word in text:
#         if word not in d:
#             d[word] = 1
#         else:
#             d[word] += 1
#     return d
#
# def stem_tokens(tokens, stemmer):
#     stemmed = []
#     for item in tokens:
#         stemmed.append(stemmer.stem(item))
#     return stemmed
#
# def tokenize(text):
#     tokens = nltk.word_tokenize(text)
#     stemmer=PorterStemmer()
#     stems = stem_tokens(tokens, stemmer)
#     return stems

def word_frequencies(trainlist,testlist,path):

    train_list = ""
    test_list = ""

    for word in trainlist:
        train_list +=word + " "

    for word in testlist:
        test_list +=word + " "

    word_freq_train = Counter(train_list.split()).items()
    word_freq_test= Counter(test_list.split()).items()

    word_freq_train = sorted(word_freq_train, key=lambda x: x[1],reverse=True)
    word_freq_test = sorted(word_freq_test, key=lambda x: x[1],reverse=True)

    N = 25
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35       # the width of the bars


    # word_freq_train = word_freq_train.most_common()
    word_freq_train_pct = map(itemgetter(1), tupleCounts2Percents(word_freq_train))[:N]
    # word_freq_train_pct = ['{:.1%}'.format(item)for item in word_freq_train_pct]

    # word_freq_test = word_freq_test.most_common()
    word_freq_test_pct = map(itemgetter(1), tupleCounts2Percents(word_freq_test))[:N]


    plt.figure()


    fig, ax = plt.subplots()

    words_train = [x[0] for x in word_freq_train][:N]
    values_train = [int(x[1]) for x in word_freq_train][:N]

    words_test = [x[0] for x in word_freq_test][:N]
    values_test = [int(x[1]) for x in word_freq_test][:N]


    plot_margin = 0.1
    x0, x1, y0, y1 = plt.axis()
    plt.axis((x0,
              x1,
              y0,
              y1*plot_margin))

    rects1 = ax.bar(ind, word_freq_train_pct, width,color='r')

    rects2 = ax.bar(ind + width, word_freq_test_pct, width, color='y')

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Words')
    # ax.set_title('Word Frequencies by Test and Training Set')
    ax.set_xticks(ind,minor=False)
    ax.set_xticks(ind + width,minor=True)
    ax.set_xticklabels(words_train,rotation=90,minor=False,ha='left')
    ax.set_xticklabels(words_test,rotation=90,minor=True,ha='left')
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:1.1f}%'.format(x*100) for x in vals])
    fig.tight_layout()

    ax.legend((rects1[0], rects2[0]), ('Train', 'Test'))

    plt.savefig(path)
    plt.clf()


def find_ngrams(input_list, n):
  return zip(*[input_list[i:] for i in range(n)])

def sentence_to_words(sentence, remove_stopwords=False):
    letters_only = re.sub('[^a-zA-Z| LOCATION_SLOT | NUMBER_SLOT]', " ", sentence)
    words = letters_only.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return (words)

def training_features(inputSentences):
    global vectorizer

    train_wordlist = []

    train_gramlist = []

    train_bigram_list = []

    train_wordbigram_list = []

    train_property_labels = []
    closed_train_property_labels = []

    for i, sentence in enumerate(inputSentences):
        words = (" ").join(sentence_to_words(sentence['parsedSentence'], True))
        train_wordlist.append(words)

        word_list = sentence_to_words(sentence['parsedSentence'], True)

        bigrams = ""
        if "depPath" in sentence.keys():
            bigrams = [("+").join(bigram).encode('utf-8') for bigram in sentence['depPath']]
            bigrams = (' ').join(map(str, bigrams))

        # print "Train bigrams are",bigrams

        train_bigram_list.append(bigrams)

        train_wordbigram_list.append(words + " " + bigrams.decode("utf-8"))

        wordgrams = find_ngrams(word_list,2)
        for i,grams in enumerate(wordgrams):
          wordgrams[i] = '+'.join(grams)
        wordgrams= (" ").join(wordgrams)
        train_gramlist.append(wordgrams)

        train_property_labels.append(sentence['predictedPropertyOpen'])
        closed_train_property_labels.append(sentence['predictedPropertyClosed'])

    return train_wordlist, train_gramlist, train_bigram_list, train_wordbigram_list, train_property_labels,closed_train_property_labels

def test_features(inputSentences):
    global vectorizer

    test_wordlist = []

    test_gramlist = []

    test_bigram_list = []

    test_wordbigram_list = []

    test_property_labels = []

    for sentence in inputSentences:
        words = " ".join(sentence_to_words(sentence['parsedSentence'], True))
        test_wordlist.append(words)

        word_list = sentence_to_words(sentence['parsedSentence'], True)

        wordgrams = find_ngrams(word_list,2)
        for i,grams in enumerate(wordgrams):
          wordgrams[i] = '+'.join(grams)
        wordgrams= (" ").join(wordgrams)
        test_gramlist.append(wordgrams)

        test_property_labels.append(sentence['property'])

        bigrams = ""
        if "depPath" in sentence.keys():
            bigrams = sentence['depPath'].encode('utf-8')

        # print "Test bigrams are",bigrams

        test_bigram_list.append(bigrams)
        test_wordbigram_list.append((words + " " + bigrams.decode("utf-8")))

    return test_wordlist, test_gramlist, test_bigram_list, test_wordbigram_list, test_property_labels


def tupleCounts2Percents(inputList):
    total = sum(x[1] for x in inputList)*1.0
    return [(x[0], 1.*x[1]/total) for x in inputList]

def autolabel(rects,labels):
    # attach some text labels
    for i,(rect,label) in enumerate(zip(rects,labels)):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                label,
                ha='left', va='bottom',fontsize=8,rotation=45)
        # style='italic'


def loadMatrix(jsonFile):
    print "loading from file " + jsonFile
    with open(jsonFile) as freebaseFile:
        property2region2value = json.loads(freebaseFile.read())

    regions = set([])
    valueCounter = 0
    kbValues = []
    for country, property2value in property2region2value.items():
        # Check for nan values and remove them
        for property, value in property2value.items():
            if not np.isfinite(value):
                del property2value[property]
                print "REMOVED:", value, " for ", property, " ", country
        if len(property2value) == 0:
            del property2region2value[property]
            print "REMOVED property:", property, " no values left"
        else:
            # print "Length of properties is",len(property2value)
            valueCounter += len(property2value)
            regions = regions.union(set(property2value.keys()))

    for country, property2value in property2region2value.items():
        for property, value in property2value.items():
            kbValues.append(value)

    print len(kbValues),  " unique values"
    print len(property2region2value), " countries"
    print len(regions),  " unique properties"
    print valueCounter, " values loaded"
    return kbValues

def trainingTestComparison(training,oldtest,newtest,filename):
    # This both sorts out the features and the training labels
    trainingValues = []
    oldTestValues = []
    newTestValues = []

    for dict in training:
        trainingValues.append(expit(dict['location-value-pair'].values()))

    for dict in oldtest:
        oldTestValues.append(expit(dict['location-value-pair'].values()))

    for dict in newtest:
        newTestValues.append(expit(dict['location-value-pair'].values()))

    trainingValues = np.array(trainingValues).flatten()
    oldTestValues = np.array(oldTestValues).flatten()
    newTestValues = np.array(newTestValues).flatten()

    plt.figure()

    fig, ax = plt.subplots()

    ax.hist(oldTestValues,alpha=0.5,color='r',bins=25,normed=True,label='Old Test Values')  # plt.hist passes it's arguments to np.histogram
    ax.hist(trainingValues,alpha=0.5,color='y',bins=25,normed=True,label='Training Values')
    ax.hist(newTestValues,alpha=0.5,color='b',bins=25,normed=True,label='New Test Values')  # plt.hist passes it's arguments to np.histogram
    plt.title("Histogram of Values")
    ax.legend(loc='upper right')
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:1.1f}%'.format(x) for x in vals])
    plt.savefig(filename)
    plt.clf()


def trainingKBComparison(kbValues,input_sentences,filename):
    # This both sorts out the features and the training labels
    trainingValues = []

    for dict in input_sentences:
        trainingValues.append(expit(dict['location-value-pair'].values()))

    trainingValues = np.array(trainingValues).flatten()

    # Load the KB
    plt.figure()

    fig, ax = plt.subplots()

    for i,value in enumerate(kbValues):
        kbValues[i] = expit(value)

    ax.hist(kbValues,alpha=0.5,color='g',normed=True,bins=25,label='KB Values')  # plt.hist passes it's arguments to np.histogram
    ax.hist(trainingValues,alpha=0.5,color='r',normed=True,bins=25,label='Training Values')
    plt.title("Histogram of Values")
    ax.legend(loc='upper right')
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:1.1f}%'.format(x) for x in vals])
    plt.savefig(filename)
    plt.clf()

def countryChartList(inputlist,path):
    seen_countries = Counter()

    for dict in inputlist:
        seen_countries += Counter(dict['location-value-pair'].keys())

    seen_countries = seen_countries.most_common()[:25]

    seen_countries_percentage = map(itemgetter(1), tupleCounts2Percents(seen_countries))
    seen_countries_percentage = ['{:.1%}'.format(item)for item in seen_countries_percentage]

    yvals = map(itemgetter(1), seen_countries)
    xvals = map(itemgetter(0), seen_countries)

    plt.figure()
    countrychart = plt.bar(range(len(seen_countries)), yvals, width=0.9,alpha=0.6)
    plt.xticks(range(len(seen_countries)), xvals,rotation=90,ha='left')

    plot_margin = 1.15
    x0, x1, y0, y1 = plt.axis()
    plt.axis((x0,
              x1,
              y0,
              y1*plot_margin))

    plt.ylabel('Occurrences')

    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.tick_params(axis='both', which='minor', labelsize=8)
    plt.tight_layout()

    autolabel(countrychart,seen_countries_percentage)

    plt.savefig(path)
    plt.clf()

def valueChartList(inputlist,path):
    seen_values = Counter()

    for dict in inputlist:
        seen_values += Counter(dict['location-value-pair'].values())

    seen_values = seen_values.most_common()[:25]
    seen_values_pct = map(itemgetter(1), tupleCounts2Percents(seen_values))
    seen_values_pct = ['{:.1%}'.format(item)for item in seen_values_pct]

    plt.figure()
    numberchart = plt.bar(range(len(seen_values)), map(itemgetter(1), seen_values), width=0.9,alpha=0.6)
    plt.xticks(range(len(seen_values)), map(itemgetter(0), seen_values),ha='left')

    plt.ylabel('Occurrences')

    plot_margin = 1.15
    x0, x1, y0, y1 = plt.axis()
    plt.axis((x0,
              x1,
              y0,
              y1*plot_margin))

    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.tick_params(axis='both', which='minor', labelsize=8)
    plt.tight_layout()

    autolabel(numberchart,seen_values_pct)

    plt.savefig(path)
    plt.clf()

def chartProperties(counter,path):

    seen_properties = sorted(counter, key=lambda x: x[1],reverse=True)
    seen_values_pct = map(itemgetter(1), tupleCounts2Percents(seen_properties))
    seen_values_pct = ['{:.1%}'.format(item)for item in seen_values_pct]

    plt.figure()

    numberchart = plt.bar(range(len(seen_properties)), map(itemgetter(1), seen_properties), width=0.9,alpha=0.6)
    plt.xticks(range(len(seen_properties)), map(itemgetter(0), seen_properties),rotation=90,ha='left')

    plt.ylabel('Occurrences')

    plot_margin = 1.15
    x0, x1, y0, y1 = plt.axis()
    plt.axis((x0,
              x1,
              y0,
              y1*plot_margin))

    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.tick_params(axis='both', which='minor', labelsize=8)
    plt.tight_layout()

    autolabel(numberchart,seen_values_pct)

    plt.savefig(path)
    plt.clf()

def distantSupervisionChart(inputlist,masterPath):

    open_properties = ""
    closed_properties = ""

    for dict in inputlist:
        open_properties +=dict['predictedPropertyOpen'].split('/')[3] + " "
        closed_properties+=dict['predictedPropertyClosed'].split('/')[3] + " "

    seen_open_labels = Counter(open_properties.split()).items()
    seen_closed_labels = Counter(closed_properties.split()).items()

    chartProperties(seen_open_labels,os.path.join(masterPath,'distantSupervisionOpen.png'))
    chartProperties(seen_closed_labels,os.path.join(masterPath,'distantSupervisionClosed.png'))

def propertyChartList(inputlist,path):


    properties = ""

    for dict in inputlist:
        properties +=dict['property'].split('/')[3] + " "

    seen_properties = Counter(properties.split()).items()

    chartProperties(seen_properties,path)


'''
Load in all the values
/Users/dhruv/Documents/university/ClaimDetection/data/freebaseTriples.json
/Users/dhruv/Documents/university/ClaimDetection/data/output/predictedPropertiesZero.json
/Users/dhruv/Documents/university/ClaimDetection/data/output/fullTestLabels.json
/Users/dhruv/Documents/university/ClaimDetection/data/output/cleanFullLabels.json
/Users/dhruv/Documents/university/ClaimDetection/data/output/mainMatrixFiltered.json
/Users/dhruv/Documents/university/ClaimDetection/data/output/theMatrixExtend120TokenFiltered_2_2_0.1_0.5_fixed2.json
/Users/dhruv/Documents/university/ClaimDetection/data/output/zero/data_analysis/

'''


if __name__ == "__main__":

    # Load in the values
    kbNumbers = np.array(loadMatrix(sys.argv[1]))

    # Load the unfiltered sentences for labelling
    with open(sys.argv[2]) as trainingSentences:
        trainingSentences = json.loads(trainingSentences.read())

    print "Length of trainingSentences is",len(trainingSentences)

     # Load the test set from Andreas
    with open(sys.argv[3]) as oldTestSentences:
        oldTestSentences = json.loads(oldTestSentences.read())

    print "Length of oldTestSentences is",len(oldTestSentences)

    # Load the full clean test set
    with open(sys.argv[4]) as newTestSentences:
        newTestSentences = json.loads(newTestSentences.read())

    print "Length of newTestSentences is",len(newTestSentences)

    # Load the post-filtering Andreas patterns
    with open(sys.argv[5]) as newPatterns:
        newPatterns = json.loads(newPatterns.read())

    print "Length of newPatterns is",len(newPatterns)

    # Load the old Andreas patterns
    with open(sys.argv[6]) as oldPatterns:
        oldPatterns = json.loads(oldPatterns.read())

    print "Length of oldPatterns is",len(oldPatterns)

    # Store the figures directory
    figures = sys.argv[7]

    print "All data has been loaded.\n"

    plt.close('all')

    print "Comparing KB values to training values...\n"
    # Now do a comparison between KB and Training Sentences
    trainingKBComparison(kbNumbers,trainingSentences,os.path.join(figures,'kbComparison.png'))

    print "Comparing test values to training values...\n"
    # Now do a comparison between KB and Training Sentences
    trainingTestComparison(trainingSentences,oldTestSentences,newTestSentences,os.path.join(figures,'trainTestComparison.png'))

    print "Starting training country charting...\n"

    countryChartList(trainingSentences,os.path.join(figures,'trainingSentenceCountries.png'))

    # print "Starting histogram of training...\n"
    #
    # valueChartList(trainingSentences,os.path.join(figures,'trainingSentenceValues.png'))

    print "Starting training property charting...\n"

    distantSupervisionChart(trainingSentences,figures)

    print "Starting old test country charting...\n"
    countryChartList(oldTestSentences,os.path.join(figures,'oldTestCountries.png'))

    # print "Starting old test value charting...\n"
    #
    # valueChartList(oldTestSentences,os.path.join(figures,'oldTestValues.png'))
    #
    print "Starting old test property charting...\n"

    propertyChartList(oldTestSentences,os.path.join(figures,'oldTestProperties.png'))

    print "Starting new test country charting...\n"

    countryChartList(newTestSentences,os.path.join(figures,'newTestCountries.png'))

    # print "Starting new test value charting...\n"
    #
    # valueChartList(newTestSentences,os.path.join(figures,'newTestValues.png'))

    print "Starting new test property charting...\n"

    propertyChartList(newTestSentences,os.path.join(figures,'newTestProperties.png'))

    '''
    Define the pipeline
    '''
    vectorizer = Pipeline([('vect', CountVectorizer(analyzer="word",token_pattern="[\S]+",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),('tfidf', TfidfTransformer(use_idf=True,norm='l2',sublinear_tf=True)),
                        ])

    print "Getting all the words in the training sentences...\n"
    train_wordlist, train_gramlist, train_bigram_list, train_wordbigram_list, train_property_labels,closed_train_property_labels = training_features(trainingSentences)

    print "Getting all the words in the old test sentences...\n"
    old_test_words, old_test_bigrams, old_test_depgrams, old_test_wordgrams, old_test_labels = test_features(oldTestSentences)

    print "Getting all the words in the new test sentences...\n"
    new_test_words, new_test_bigrams, new_test_depgrams, new_test_wordgrams, new_test_labels = test_features(newTestSentences)


    print "Plotting term frequencies for old words...\n"
    word_frequencies(train_wordlist, old_test_words,os.path.join(figures,'wordFreqOldWords.png'))

    print "Plotting term frequencies for old bigrams...\n"

    word_frequencies(train_gramlist, old_test_bigrams,os.path.join(figures,'wordFreqOldBigrams.png'))

    print "Plotting term frequencies for old wordgrams...\n"

    word_frequencies(train_bigram_list, old_test_depgrams,os.path.join(figures,'wordFreqOldDepgrams.png'))

    print "Plotting term frequencies for new words"
    word_frequencies(train_wordlist, new_test_words,os.path.join(figures,'wordFreqNewWords.png'))

    print "Plotting term frequencies for new depgrams...\n"

    word_frequencies(train_gramlist, new_test_bigrams,os.path.join(figures,'wordFreqNewBigrams.png'))

    print "Plotting term frequencies for new wordgrams...\n"

    word_frequencies(train_bigram_list, new_test_depgrams,os.path.join(figures,'wordFreqNewDepgrams.png'))