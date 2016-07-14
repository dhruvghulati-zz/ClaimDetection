import re
from nltk.corpus import stopwords  # Import the stop word list
import json
import numpy as np
import pandas as pd
import sys
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
    # 2. Remove non-letters, and ensure location and number slots not split
    # a-zA-Z
    letters_only = re.sub('[^a-zA-Z| LOCATION_SLOT | NUMBER_SLOT]', " ", sentence)
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
            if sentence['predictedPropertyOpenThreshold'] != "no_region":
                binary_train_labels.append(1)
            else:
                binary_train_labels.append(0)
            # Closed evaluation only include certain training sentences
            closed_train_wordlist.append(" ".join(sentence_to_words(sentence['parsedSentence'], True)))
            closed_train_property_labels.append(sentence['predictedPropertyClosed'])
            closed_train_property_labels_threshold.append(sentence['predictedPropertyClosedThreshold'])
            if sentence['predictedPropertyClosedThreshold'] != "no_region":
                binary_closed_train_labels.append(1)
            else:
                binary_closed_train_labels.append(0)
    # print "These are the clean words in the training sentences: ", train_wordlist
    # print "These are the labels in the training sentences: ", train_labels
    print "Creating the bag of words...\n"
    train_data_features = vectorizer.fit_transform(train_wordlist)
    closed_train_data_features = vectorizer.fit_transform(closed_train_wordlist)

    train_data_features = train_data_features.toarray()
    closed_train_data_features = closed_train_data_features.toarray()

    return train_data_features, closed_train_data_features

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


def test_features(testSentences):
    global vectorizer

    for sentence in testSentences:
        if sentence['parsedSentence'] != {} and sentence['mape_label'] != {}:
            clean_test_sentences.append(" ".join(sentence_to_words(sentence['parsedSentence'], True)))
            # print clean_test_sentences
            binary_test_labels.append(sentence['mape_label'])
            test_property_labels.append(sentence['property'])

    # print "These are the clean words in the test sentences: ", clean_test_sentences
    # print "These are the mape labels in the test sentences: ", binary_test_labels
    test_data_features = vectorizer.transform(clean_test_sentences)
    test_data_features = test_data_features.toarray()

    return test_data_features


def win(row):
    if row['truePropLabels'] == row['predictedPropLabels'] and row['trueClaimLabels'] == 1:
        val = 1
    elif row['truePropLabels'] != row['predictedPropLabels'] and row['trueClaimLabels'] == 0:
        val = 1
    else:
        val = 0
    return val


def propEvaluation(truePropLabels, predictedPropLabels, trueClaimLabels):
    precision = 0
    # TODO - need to create a per property chart
    # print type(truePropLabels)
    # print type(predictedPropLabels)
    # print type(trueClaimLabels)
    #
    # print truePropLabels
    # print predictedPropLabels
    # print trueClaimLabels

    summaryDF = pd.DataFrame(({'truePropLabels': truePropLabels, 'trueClaimLabels': trueClaimLabels,
                               'predictedPropLabels': predictedPropLabels}))

    summaryDF['modelWin'] = summaryDF.apply(win, axis=1)

    # summaryDF = summaryDF.groupby('truePropLabels').sum()
    #
    # summaryDF = summaryDF.append(summaryDF.sum(numeric_only=True), ignore_index=True)

    # .groupby('truePropLabels').count()

    print "Grouped dataframe by property is\n"

    print summaryDF.head(n=5)

    # for trueProp, trueClaim, predPropLabel in zip(truePropLabels, trueClaimLabels, predictedPropLabels):
    #     # print(trueProp, trueClaim, predPropLabel)
    #     if ((trueProp==predPropLabel) and (trueClaim==1)) or ((trueProp!=predPropLabel) and (trueClaim==0)):
    #         precision+=1

    # precisionPercent = precision/len(truePropLabels)
    #
    # print "Precision of prediction is", precisionPercent

    return precision

    # return precisionPercent


# /Users/dhruv/Documents/university/ClaimDetection/data/output/predictedProperties.json
# /Users/dhruv/Documents/university/ClaimDetection/data/output/hyperTestLabels.json
# /Users/dhruv/Documents/university/ClaimDetection/data/regressionResult.json
# python src/main/logisticBagOfWords.py data/output/predictedProperties.json data/output/hyperTestLabels.json data/regressionResult.json data/featuresKept.json
# if __name__ == "__main__":
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

# y_multi_true = le.transform(test['property'])
# print "These are the true classes", y_multi_true

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

# # print "Training data features are: ", train_data_features
#
# This both sorts out the features and the training labels
train_data_features, closed_train_data_features = training_features(pattern2regions)

print len(train_data_features), "sets of open training features"
print len(closed_train_data_features), "sets of closed training features"

print "There are ", binary_train_labels.count(1), "positive mape labels"
print "There are ", binary_train_labels.count(0), "negative mape labels"
print "There are ", binary_closed_train_labels.count(1), "positive closed mape labels"
print "There are ", binary_closed_train_labels.count(0), "negative closed mape labels\n"

# binary_train_data_features, binary_train_labels = balanced_subsample(train_data_features,binary_train_labels)

# train_data_features, train_property_labels  = balanced_subsample(train_data_features,train_property_labels)

# print binary_train_labels

# print "There are ",binary_train_labels.count(1), "positive mape labels"
# print "There are ",binary_train_labels.count(0), "negative mape labels"

# print len(train_data_features),"sets of training features"

# Initialize a Logistic Regression on the statistical region
# binary_logit = LogisticRegression(fit_intercept=True, class_weight='auto')

multi_logit = LogisticRegression(fit_intercept=True, class_weight='auto', multi_class='multinomial', solver='newton-cg')

# # Fit the logistic classifiers to the training set, using the bag of words as
# # features and the sentiment labels as the response variable
# #
# print "Fitting the open binary logistic regression model...\n"
# # This may take a few minutes to run
# # This seems to be wrong, based on empty training labels
# binary_logit = binary_logit.fit(train_data_features, binary_train_labels)
#
# print "Fitting the closed binary logistic regression model...\n"
#
# binary_closed_logit = binary_logit.fit(closed_train_data_features, binary_closed_train_labels)
#
# print "These are the training labels\n"
#
# print train_property_labels
# Note, these are the same but only because the threshold actually removed totally some properties e.g. foreign_direct_investment_net_inflows, but added things like no_region
print "There are ", len(set(train_property_labels)), "open training classes"
print "There are ", len(set(train_property_labels_threshold)), "open training classes w/threshold"
print "There are ", len(set(closed_train_property_labels)), "closed training properties"
print "There are ", len(set(closed_train_property_labels_threshold)), "closed training properties w/ threshold"

print "Fitting the open multinomial logistic regression model w/ MAPE threshold...\n"

# le_train = le.fit(train_property_labels)
# train_classes = le_train.classes_


# train_property_labels = le.transform(train_property_labels)

open_threshold_multi_logit = multi_logit.fit(train_data_features, train_property_labels_threshold)
print "Fitting the open multinomial logistic regression model without MAPE threshold...\n"
open_multi_logit = multi_logit.fit(train_data_features, train_property_labels)
print "Fitting the closed multinomial logistic regression model without MAPE threshold...\n"
closed_multi_logit = multi_logit.fit(closed_train_data_features, closed_train_property_labels)
print "Fitting the closed multinomial logistic regression model with MAPE threshold...\n"
closed_threshold_multi_logit = multi_logit.fit(closed_train_data_features, closed_train_property_labels_threshold)

# Create an empty list and append the clean reviews one by one
clean_test_sentences = []

print "Cleaning and parsing the test set ...\n"

print "Get a bag of words for the test set, and convert to a numpy array\n"

test_data_features = test_features(finalTestSentences)

print len(test_data_features), "sets of test features"

print test_data_features

# print "There are ", binary_test_labels.count(1), "positive mape labels"
# print "There are ", binary_test_labels.count(0), "negative mape labels"
#
# # Use the logistic regression to make predictions
# print "Predicting binary test labels...\n"
#
# binary_logit_result = binary_logit.predict(test_data_features)

print "Predicting open multinomial test labels without threshold...\n"

y_multi_logit_result_open = np.array(open_multi_logit.predict(test_data_features))

print "Predicting open multinomial test labels w/ threshold...\n"

y_multi_logit_result_open_threshold = np.array(open_threshold_multi_logit.predict(test_data_features))

print "Predicting closed multinomial test labels w/ threshold...\n"

y_multi_logit_result_closed = np.array(closed_multi_logit.predict(test_data_features))

print "Predicting open multinomial test labels without threshold...\n"

y_multi_logit_result_closed_threshold = np.array(closed_threshold_multi_logit.predict(test_data_features))

# TODO - this may not be relevant any more
# Convert the multi logit result to binary evaluation
# binary_multi_logit_result = []
# for result in y_multi_logit_result_open_threshold:
#     if result == "no_region":
#         binary_multi_logit_result.append(0)
#     else:
#         binary_multi_logit_result.append(1)

# Load in the test data
test = pd.DataFrame(finalTestSentences)

threshold = test['threshold'][0]
# print "These are the property predictions\n"
#
# print multi_logit_result

# TODO - Need to clean these up


# These are the ground truths
y_multi_true = np.array(test['property'])
y_true_claim = np.array(test['claim'])

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
closed_categorical_random_threshold = rng.choice(list(closed_unique_train_labels_threshold), len(finalTestSentences))
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

# TODO - this should be cleaner code
# Convert the categorical predictions to binary based on if matching property
for open_sv_property, closed_sv_property, open_threshold_sv_property, closed_threshold_sv_property, open_multinomial_property, open_threshold_multinomial_property, closed_multinomial_property, closed_threshold_multinomial_property, cat_random, cat_random_threshold,\
    closed_cat_random,\
    closed_cat_random_threshold,\
    true_property in zip(y_distant_sv_property_open, y_distant_sv_property_closed,y_distant_sv_property_openThreshold,
            y_distant_sv_property_closedThreshold, y_multi_logit_result_open, y_multi_logit_result_open_threshold,
            y_multi_logit_result_closed, y_multi_logit_result_closed_threshold,
            categorical_random,
            categorical_random_threshold,
            closed_categorical_random,
            closed_categorical_random_threshold,
            y_multi_true):
    if open_sv_property == true_property:
        y_open_distant_sv_to_binary.append(1)
    else:
        y_open_distant_sv_to_binary.append(0)
    if closed_sv_property == true_property:
        y_closed_distant_sv_to_binary.append(1)
    else:
        y_closed_distant_sv_to_binary.append(0)
    if open_threshold_sv_property == true_property:
        y_openThreshold_distant_sv_to_binary.append(1)
    else:
        y_openThreshold_distant_sv_to_binary.append(0)
    if closed_threshold_sv_property == true_property:
        y_closedThreshold_distant_sv_to_binary.append(1)
    else:
        y_closedThreshold_distant_sv_to_binary.append(0)
    if open_multinomial_property == true_property:
        y_multi_logit_result_open_binary.append(1)
    else:
        y_multi_logit_result_open_binary.append(0)
    if open_threshold_multinomial_property == true_property:
        y_multi_logit_result_open_threshold_binary.append(1)
    else:
        y_multi_logit_result_open_threshold_binary.append(0)
    if closed_multinomial_property == true_property:
        y_multi_logit_result_closed_binary.append(1)
    else:
        y_multi_logit_result_closed_binary.append(0)
    if closed_threshold_multinomial_property == true_property:
        y_multi_logit_result_closed_threshold_binary.append(1)
    else:
        y_multi_logit_result_closed_threshold_binary.append(0)
    if cat_random == true_property:
        y_cat_random_to_binary.append(1)
    else:
        y_cat_random_to_binary.append(0)
    if cat_random_threshold == true_property:
        y_cat_random_to_binary_threshold.append(1)
    else:
        y_cat_random_to_binary_threshold.append(0)
    if closed_cat_random == true_property:
        y_closed_random_to_binary.append(1)
    else:
        y_closed_random_to_binary.append(0)
    if closed_cat_random_threshold == true_property:
        y_closedCat_random_to_binary_threshold.append(1)
    else:
        y_closedCat_random_to_binary_threshold.append(0)


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
# This was an issue on command line
testSet = str(os.path.splitext(sys.argv[2])[0]).split("/")[8]
# testSet = str(os.path.splitext(sys.argv[2])[0]).split("/")[2]

resultPath = os.path.join(sys.argv[5] + "test/" + testSet + '_' + str(threshold) + '_regressionResult.csv')

output.to_csv(path_or_buf=resultPath, encoding='utf-8')

# Now we write our precision F1 etc to an Excel file
# TODO - this code needs to be more scalable
random_precision = precision_score(y_true_claim, y_randpred)
random_recall = recall_score(y_true_claim, y_randpred)
random_f1 = f1_score(y_true_claim, y_randpred)

random_data = {'precision': random_precision,
               'recall': random_recall,
               'f1': random_f1}

randomDF = pd.DataFrame(random_data, index=['Binary Random Baseline'])

cat_random_precision = precision_score(y_true_claim, y_cat_random_to_binary)
cat_random_recall = recall_score(y_true_claim, y_cat_random_to_binary)
cat_random_f1 = f1_score(y_true_claim, y_cat_random_to_binary)

cat_random_data = {'precision': cat_random_precision,
                   'recall': cat_random_recall,
                   'f1': cat_random_f1}

cat_randomDF = pd.DataFrame(cat_random_data, index=['Categorical Random Baseline'])

cat_random_precision_threshold = precision_score(y_true_claim, y_cat_random_to_binary_threshold)
cat_random_recall_threshold = recall_score(y_true_claim, y_cat_random_to_binary_threshold)
cat_random_f1_threshold = f1_score(y_true_claim, y_cat_random_to_binary_threshold)

cat_random_data_threshold = {'precision': cat_random_precision_threshold,
               'recall': cat_random_recall_threshold,
               'f1': cat_random_f1_threshold}

cat_randomDF_threshold = pd.DataFrame(cat_random_data_threshold, index=['Categorical Random Baseline w/ Threshold'])

closedCat_random_precision = precision_score(y_true_claim, y_closed_random_to_binary)
closedCat_random_recall = recall_score(y_true_claim, y_closed_random_to_binary)
closedCat_random_f1 = f1_score(y_true_claim, y_closed_random_to_binary)

closedCat_random_data = {'precision': closedCat_random_precision,
               'recall': closedCat_random_recall,
               'f1': closedCat_random_f1}

closedCat_randomDF = pd.DataFrame(closedCat_random_data, index=['Closed Category Random Baseline'])

closedCat_random_precision_threshold = precision_score(y_true_claim, y_closedCat_random_to_binary_threshold)
closedCat_random_recall_threshold = recall_score(y_true_claim, y_closedCat_random_to_binary_threshold)
closedCat_random_f1_threshold = f1_score(y_true_claim, y_closedCat_random_to_binary_threshold)

closedCat_random_data_threshold = {'precision': closedCat_random_precision_threshold,
               'recall': closedCat_random_recall_threshold,
               'f1': closedCat_random_f1_threshold}

closedCat_randomDF_threshold = pd.DataFrame(closedCat_random_data_threshold, index=['Closed Category Random Baseline w/ Threshold'])

andreas_precision = precision_score(y_true_claim, y_pospred)
andreas_recall = recall_score(y_true_claim, y_pospred)
andreas_f1 = f1_score(y_true_claim, y_pospred)

andreas_data = {'precision': andreas_precision,
                'recall': andreas_recall,
                'f1': andreas_f1}

andreasDF = pd.DataFrame(andreas_data, index=['Andreas Model (Previous Model)'])

neg_precision = precision_score(y_true_claim, y_negpred)
neg_recall = recall_score(y_true_claim, y_negpred)
neg_f1 = f1_score(y_true_claim, y_negpred)

neg_data = {'precision': neg_precision,
            'recall': neg_recall,
            'f1': neg_f1}

negDF = pd.DataFrame(neg_data, index=['Negative Naive Baseline'])

open_mape_rule_precision = precision_score(y_true_claim, y_open_distant_sv_to_binary)
open_mape_rule_recall = recall_score(y_true_claim, y_open_distant_sv_to_binary)
open_mape_rule_f1 = f1_score(y_true_claim, y_open_distant_sv_to_binary)

open_mape_rule_data = {'precision': open_mape_rule_precision,
                       'recall': open_mape_rule_recall,
                       'f1': open_mape_rule_f1}

open_mapeDF = pd.DataFrame(open_mape_rule_data, index=['Open Category Distant Supervision Model'])

openThreshold_mape_rule_precision = precision_score(y_true_claim, y_openThreshold_distant_sv_to_binary)
openThreshold_mape_rule_recall = recall_score(y_true_claim, y_openThreshold_distant_sv_to_binary)
openThreshold_mape_rule_f1 = f1_score(y_true_claim, y_openThreshold_distant_sv_to_binary)

openThreshold_mape_rule_data = {'precision': openThreshold_mape_rule_precision,
                                'recall': openThreshold_mape_rule_recall,
                                'f1': openThreshold_mape_rule_f1}

openThreshold_mapeDF = pd.DataFrame(openThreshold_mape_rule_data,
                                    index=['Open Category Rule Based MAPE Threshold Distant Supervision Model'])

closed_mape_rule_precision = precision_score(y_true_claim, y_closed_distant_sv_to_binary)
closed_mape_rule_recall = recall_score(y_true_claim, y_closed_distant_sv_to_binary)
closed_mape_rule_f1 = f1_score(y_true_claim, y_closed_distant_sv_to_binary)

closed_mape_rule_data = {'precision': closed_mape_rule_precision,
                         'recall': closed_mape_rule_recall,
                         'f1': closed_mape_rule_f1}

closed_mapeDF = pd.DataFrame(closed_mape_rule_data, index=['Closed Category Distant Supervision Model'])

closedThreshold_mape_rule_precision = precision_score(y_true_claim, y_closedThreshold_distant_sv_to_binary)
closedThreshold_mape_rule_recall = recall_score(y_true_claim, y_closedThreshold_distant_sv_to_binary)
closedThreshold_mape_rule_f1 = f1_score(y_true_claim, y_closedThreshold_distant_sv_to_binary)

closedThreshold_mape_rule_data = {'precision': closedThreshold_mape_rule_precision,
                                  'recall': closedThreshold_mape_rule_recall,
                                  'f1': closedThreshold_mape_rule_f1}

closedThreshold_mapeDF = pd.DataFrame(closedThreshold_mape_rule_data,
                                      index=['Closed Category Rule Based MAPE Threshold Distant Supervision Model'])

open_multi_precision = precision_score(y_true_claim, y_multi_logit_result_open_binary)
open_multi_recall = recall_score(y_true_claim, y_multi_logit_result_open_binary)
open_multi_f1 = f1_score(y_true_claim, y_multi_logit_result_open_binary)

open_multi_data = {'precision': open_multi_precision,
                   'recall': open_multi_recall,
                   'f1': open_multi_f1}

open_multiDF = pd.DataFrame(open_multi_data,
                            index=['Open Category Multinomial Logistic Regression w/ Binary Evaluation (Bag of Words)'])

openThreshold_multi_precision = precision_score(y_true_claim, y_multi_logit_result_open_threshold_binary)
openThreshold_multi_recall = recall_score(y_true_claim, y_multi_logit_result_open_threshold_binary)
openThreshold_multi_f1 = f1_score(y_true_claim, y_multi_logit_result_open_threshold_binary)

openThreshold_multi_data = {'precision': openThreshold_multi_precision,
                            'recall': openThreshold_multi_recall,
                            'f1': openThreshold_multi_f1}

openThreshold_multiDF = pd.DataFrame(openThreshold_multi_data, index=[
    'Open Category Multinomial with MAPE Threshold Logistic Regression w/ Binary Evaluation (Bag of Words)'])

closed_multi_precision = precision_score(y_true_claim, y_multi_logit_result_closed_binary)
closed_multi_recall = recall_score(y_true_claim, y_multi_logit_result_closed_binary)
closed_multi_f1 = f1_score(y_true_claim, y_multi_logit_result_closed_binary)

closed_multi_data = {'precision': closed_multi_precision,
                     'recall': closed_multi_recall,
                     'f1': closed_multi_recall}

closed_multiDF = pd.DataFrame(closed_multi_data, index=[
    'Closed Category Multinomial Logistic Regression w/ Binary Evaluation (Bag of Words)'])

closedThreshold_multi_precision = precision_score(y_true_claim, y_multi_logit_result_closed_threshold_binary)
closedThreshold_multi_recall = recall_score(y_true_claim, y_multi_logit_result_closed_threshold_binary)
closedThreshold_multi_f1 = f1_score(y_true_claim, y_multi_logit_result_closed_threshold_binary)

closedThreshold_multi_data = {'precision': closedThreshold_multi_precision,
                              'recall': closedThreshold_multi_recall,
                              'f1': closedThreshold_multi_f1}

closedThreshold_multiDF = pd.DataFrame(closedThreshold_multi_data, index=[
    'Closed Category Multinomial with MAPE Threshold Logistic Regression w/ Binary Evaluation (Bag of Words)'])

# Combine all the dataframes together to form rows
summaryDF = pd.concat([randomDF, cat_randomDF, andreasDF, negDF,
                       open_mapeDF, openThreshold_mapeDF, closed_mapeDF, closedThreshold_mapeDF,
                       open_multiDF, openThreshold_multiDF, closed_multiDF, closedThreshold_multiDF
                       ])

precisionF1Path = os.path.join(sys.argv[5] + "test/" + testSet + '_' + str(threshold) + '_summaryEval.csv')
# Change what I actually output to csv
summaryDF.to_csv(path_or_buf=precisionF1Path, encoding='utf-8')
# tsv = open(sys.argv[5], "wb")
