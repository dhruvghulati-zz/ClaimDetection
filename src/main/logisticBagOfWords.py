import re
from nltk.corpus import stopwords # Import the stop word list
import json
import numpy as np
import pandas as pd
import sys
import sklearn
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
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
    Are we training on too many positive instances (no region)?
'''

def sentence_to_words(sentence,remove_stopwords=False):
    # 2. Remove non-letters, and ensure location and number slots not split
    # a-zA-Z
    letters_only = re.sub("[^a-zA-Z| LOCATION_SLOT | NUMBER_SLOT]", " ", sentence)
    # letters_only = sentence

    # [x.strip() for x in re.findall('\s*(\w+|\W+)', line)]
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
    for i, sentence in enumerate(inputSentences[:15000]):
    # Dont train if the sentence contains a random region we don't care about
        if sentence and sentence['predictedRegion'] in properties:
                train_wordlist.append(" ".join(sentence_to_words(sentence['parsedSentence'], True)))
                if sentence['predictedRegion']!="no_region":
                    binary_train_labels.append(1)
                    train_property_labels.append(sentence['predictedRegion'])
                else:
                    binary_train_labels.append(0)
                    train_property_labels.append("no_region")
                # binary_train_labels.append(0)
                # train_property_labels.append("no_region")
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


def balanced_subsample(x,y,subsample_size=1.0):

    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []

    for ci,this_xs in class_xs:
        if len(this_xs) > use_elems:
            rng.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    return xs,ys


def test_features(testSentences):

    global vectorizer

    for sentence in testSentences:
        if sentence['parsedSentence']!={} and sentence['mape_label']!={}:
            clean_test_sentences.append(" ".join(sentence_to_words(sentence['parsedSentence'], True)))
            # print clean_test_sentences
            binary_test_labels.append(sentence['mape_label'])
            test_property_labels.append(sentence['property'])

    # print "These are the clean words in the test sentences: ", clean_test_sentences
    # print "These are the mape labels in the test sentences: ", binary_test_labels
    test_data_features = vectorizer.transform(clean_test_sentences)
    test_data_features = test_data_features.toarray()

    return test_data_features

def propEvaluation(truePropLabels,predictedPropLabels, trueClaimLabels):

    precision = 0

    for trueProp, trueClaim, predPropLabel in zip(truePropLabels, trueClaimLabels, predictedPropLabels):
        print(trueProp, trueClaim, predPropLabel)
        if (trueProp==predPropLabel) and (trueClaim==1):
            precision+=1

    precisionPercent = precision/len(truePropLabels)

    print "Precision of prediction is", precisionPercent

    return precisionPercent

def binEvaluation(truePropLabels,predictedBinLabels, trueClaimLabels):

    precision = 0

    for trueProp, trueClaim, predBinLabel in zip(truePropLabels, trueClaimLabels, predictedBinLabels):
        print(trueProp, trueClaim, predBinLabel)
        if (trueProp==predBinLabel) and (trueClaim==1):
            precision+=1

    precisionPercent = precision/len(truePropLabels)

    print "Precision of prediction is", precisionPercent

    return precisionPercent


# /Users/dhruv/Documents/university/ClaimDetection/data/output/predictedProperties.json
# /Users/dhruv/Documents/university/ClaimDetection/data/output/hyperTestLabels.json
# /Users/dhruv/Documents/university/ClaimDetection/data/regressionResult.json
# python src/main/logisticBagOfWords.py data/output/predictedProperties.json data/output/hyperTestLabels.json data/regressionResult.json data/featuresKept.json
# if __name__ == "__main__":
    # training data
    # load the sentence file for training

with open(sys.argv[1]) as trainingSentences:
    pattern2regions = json.loads(trainingSentences.read())

print "We have ", len(pattern2regions)," training sentences."
# We load in the allowable features and also no_region
with open(sys.argv[3]) as featuresKept:
    properties = json.loads(featuresKept.read())
properties.append("no_region")

with open(sys.argv[2]) as testSentences:
    testSentences = json.loads(testSentences.read())


finalTestSentences = []

for sentence in testSentences:
    if sentence['parsedSentence']!={} and sentence['mape_label']!={} and sentence['mape']!={} and sentence['property']!={} and sentence['property'] in properties:
        # print sentence['property']
        finalTestSentences.append(sentence)

# y_multi_true = le.transform(test['property'])
# print "These are the true classes", y_multi_true

train_wordlist = []
binary_train_labels = []
test_wordlist = []
binary_test_labels = []
train_property_labels = []
test_property_labels = []

vectorizer = CountVectorizer(analyzer = "word",   \
                 tokenizer = None,    \
                 preprocessor = None, \
                 stop_words = None,\
                 max_features=5000)


print "Getting all the words in the training sentences...\n"

# # print "Training data features are: ", train_data_features
#
# This both sorts out the features and the training labels
train_data_features = training_features(pattern2regions)

print len(train_data_features),"sets of training features"

print "There are ",binary_train_labels.count(1), "positive mape labels"
print "There are ",binary_train_labels.count(0), "negative mape labels"

# binary_train_data_features, binary_train_labels = balanced_subsample(train_data_features,binary_train_labels)

# train_data_features, train_property_labels  = balanced_subsample(train_data_features,train_property_labels)

# print binary_train_labels

# print "There are ",binary_train_labels.count(1), "positive mape labels"
# print "There are ",binary_train_labels.count(0), "negative mape labels"

# print len(train_data_features),"sets of training features"

# Initialize a Logistic Regression on the statistical region
binary_logit = LogisticRegression(fit_intercept=True,class_weight='auto')

multi_logit = LogisticRegression(fit_intercept=True,class_weight='auto', multi_class='multinomial', solver='newton-cg')

# Fit the logistic classifiers to the training set, using the bag of words as
# features and the sentiment labels as the response variable
#
print "Fitting the binary logistic regression model...\n"
# This may take a few minutes to run
# This seems to be wrong, based on empty training labels
binary_logit = binary_logit.fit(train_data_features, binary_train_labels)
#
# print "These are the training labels\n"
#
# print train_property_labels
print "There are ", len(set(train_property_labels)),"training classes"

print "Fitting the multinomial logistic regression model...\n"

# le_train = le.fit(train_property_labels)
# train_classes = le_train.classes_


# train_property_labels = le.transform(train_property_labels)

multi_logit = multi_logit.fit(train_data_features, train_property_labels)

# Create an empty list and append the clean reviews one by one
clean_test_sentences = []

print "Cleaning and parsing the test set ...\n"

print "Get a bag of words for the test set, and convert to a numpy array\n"

test_data_features = test_features(finalTestSentences)

print len(test_data_features),"sets of test features"

print test_data_features

print "There are ",binary_test_labels.count(1), "positive mape labels"
print "There are ",binary_test_labels.count(0), "negative mape labels"

# Use the logistic regression to make predictions
print "Predicting binary test labels...\n"

binary_logit_result = binary_logit.predict(test_data_features)

print "Predicting multinomial test labels...\n"

multi_logit_result = multi_logit.predict(test_data_features)

binary_multi_logit_result = []
# Convert the multi logit result to binary evaluation
for result in multi_logit_result:
    if result=="no_region":
        binary_multi_logit_result.append(0)
    else:
        binary_multi_logit_result.append(1)

test = pd.DataFrame(finalTestSentences)

threshold = test['threshold'][0]
# print "These are the property predictions\n"
#
# print multi_logit_result

# These are the baselines

random_result = rng.randint(2, size=len(finalTestSentences))

positive_result = np.ones(len(finalTestSentences))

negative_result = np.zeros(len(finalTestSentences))


y_multi_true = np.array(test['property'])
y_andreas_mape = test['mape_label']
y_true_claim = np.array(test['claim'])
y_logpred = binary_logit_result
y_multilogpred = np.array(multi_logit_result)
y_multilogpred_binary = binary_multi_logit_result
y_model_mape = test['predicted_mape_label']
y_randpred = random_result
y_pospred = positive_result
y_negpred = negative_result

propEvaluation(y_multi_true,y_multilogpred,y_true_claim)

binEvaluation(y_multi_true,y_model_mape,y_true_claim)


# print "True claim labels are", len(y_true_claim)
# # print y_true_claim
# print "Random prediction is",len(y_randpred)
# print y_randpred

f = open(sys.argv[4], 'w')
# print >> f, 'Filename:', filename  # or f.write('...\n')
# Use this when we need to output to a file
# precision, recall, f1 = precision_recall_fscore_support(y_true_claim, y_randpred, pos_label=None,average='macro')


print >> f, "Precision, recall and F1 and support for random naive baseline are ", precision_recall_fscore_support(y_true_claim, y_randpred, pos_label=None,average='macro'),"\n"

print >> f, "Precision, recall and F1 and support for Andreas model are ", precision_recall_fscore_support(y_pospred, y_true_claim, pos_label=None,average='macro'),"\n"

print >> f, "Precision, recall and F1 and support for rule-based MAPE model are ", precision_recall_fscore_support(y_true_claim, y_model_mape, pos_label=None,average='macro'),"\n"

print >> f,"Precision, recall and F1 and support for positive naive baseline are ", precision_recall_fscore_support(y_true_claim, y_pospred, pos_label=None,average='macro'),"\n"

print >> f,"Precision, recall and F1 and support for negative naive baseline are ", precision_recall_fscore_support(y_true_claim, y_negpred, pos_label=None,average='macro'),"\n"

print >> f,"Precision, recall and F1 and support for binary logistic regression are ",precision_recall_fscore_support(y_true_claim, y_logpred, pos_label=None,average='macro'),"\n"

# # http://scikit-learn.org/stable/modules/model_evaluation.html#multiclass-and-multilabel-classification
#
print >> f,"Precision, recall and F1 and support for multinomial logistic regression converted to binary are ", precision_recall_fscore_support(y_true_claim, y_multilogpred_binary, pos_label=None,average='macro'),"\n"

f.close()

# csv_path = open(sys.argv[3],"wb")

output = pd.DataFrame(data=dict(parsed_sentence=test['parsedSentence'], property_prediction=y_multilogpred,multinomial_binary=y_multilogpred_binary,predicted_label=binary_logit_result, random_label=random_result,previous_predicted_label=test['mape_label'],claim_label=y_true_claim,actual_property_label=test['property'],positive_baseline=y_pospred, negative_baseline=y_negpred, threshold=np.full(len(y_true_claim),threshold),predicted_mape_label = y_model_mape, features=clean_test_sentences))

# print str(os.path.splitext(sys.argv[2])[0]).split("/")
# This was an issue on command line
testSet = str(os.path.splitext(sys.argv[2])[0]).split("/")[8]
# testSet = str(os.path.splitext(sys.argv[2])[0]).split("/")[2]

resultPath = os.path.join(sys.argv[5] + "test/" + testSet + '_'+str(threshold)+'_regressionResult.csv')

output.to_csv(path_or_buf=resultPath,encoding='utf-8')

# Now we write our precision F1 etc to an Excel file

random_precision = precision_score(y_true_claim, y_randpred, pos_label=None,average='macro')
random_recall = recall_score(y_true_claim, y_randpred, pos_label=None,average='macro')
random_f1 = f1_score(y_true_claim, y_randpred, pos_label=None,average='macro')

random_data = {'precision': random_precision,
    'recall': random_recall,
    'f1': random_f1}

randomDF = pd.DataFrame(random_data,index=['Random Baseline'])

andreas_precision = precision_score(y_pospred, y_true_claim, pos_label=None,average='macro')
andreas_recall = recall_score(y_pospred, y_true_claim, pos_label=None,average='macro')
andreas_f1 = f1_score(y_pospred, y_true_claim, pos_label=None,average='macro')

andreas_data = {'precision': andreas_precision,
    'recall': andreas_recall,
    'f1': andreas_f1}

andreasDF = pd.DataFrame(andreas_data,index=['Previous Model'])

mape_rule_precision = precision_score(y_true_claim, y_model_mape, pos_label=None,average='macro')
mape_rule_recall = recall_score(y_true_claim, y_model_mape, pos_label=None,average='macro')
mape_rule_f1 = f1_score(y_true_claim, y_model_mape, pos_label=None,average='macro')

mape_rule_data = {'precision': mape_rule_precision,
    'recall': mape_rule_recall,
    'f1': mape_rule_f1}

mapeDF = pd.DataFrame(random_data,index=['Rule Based MAPE Model'])

pos_precision = precision_score(y_true_claim, y_pospred, pos_label=None,average='macro')
pos_recall = recall_score(y_true_claim, y_pospred, pos_label=None,average='macro')
pos_f1 = f1_score(y_true_claim, y_pospred, pos_label=None,average='macro')

pos_data = {'precision': pos_precision,
    'recall': pos_recall,
    'f1': pos_f1}

posDF = pd.DataFrame(pos_data,index=['Positive Naive Baseline'])

neg_precision = precision_score(y_true_claim, y_negpred, pos_label=None,average='macro')
neg_recall = recall_score(y_true_claim, y_negpred, pos_label=None,average='macro')
neg_f1 = f1_score(y_true_claim, y_negpred, pos_label=None,average='macro')

neg_data = {'precision': neg_precision,
    'recall': neg_recall,
    'f1': neg_f1}

negDF = pd.DataFrame(neg_data,index=['Negative Naive Baseline'])

binary_precision = precision_score(y_true_claim, y_logpred, pos_label=None,average='macro')
binary_recall = recall_score(y_true_claim, y_logpred, pos_label=None,average='macro')
binary_f1 = f1_score(y_true_claim, y_logpred, pos_label=None,average='macro')

binary_data = {'precision': binary_precision,
    'recall': binary_recall,
    'f1': binary_f1}

binaryDF = pd.DataFrame(binary_data,index=['Binary Logistic Regression (Bag of Words)'])

multi_precision = precision_score(y_true_claim, y_multilogpred_binary, pos_label=None,average='macro')
multi_recall = recall_score(y_true_claim, y_multilogpred_binary, pos_label=None,average='macro')
multi_f1 = f1_score(y_true_claim, y_multilogpred_binary, pos_label=None,average='macro')

multi_data = {'precision': multi_precision,
    'recall': multi_recall,
    'f1': multi_f1}

multiDF = pd.DataFrame(multi_data,index=['Multinomial Logistic Regression w/ Binary Evaluation (Bag of Words)'])

summaryDF = pd.concat([randomDF,andreasDF,mapeDF,posDF,negDF,binaryDF,multiDF])


precisionF1Path = os.path.join(sys.argv[5]+"test/"+ testSet + '_'+str(threshold)+'_summaryEval.csv')
# Change what I actually output to csv
summaryDF.to_csv(path_or_buf=precisionF1Path,encoding='utf-8')
# tsv = open(sys.argv[5], "wb")



