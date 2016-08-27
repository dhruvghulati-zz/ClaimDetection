import math
import re
from nltk.corpus import stopwords  # Import the stop word list
import json
import numpy as np
import copy
import sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import os
from sklearn.pipeline import Pipeline
from collections import defaultdict
xs = defaultdict(list)
from numpy import inf


# This is about loading any file with property: region:value format
# def loadMatrix(jsonFile):
#     print "loading from file " + jsonFile
#     with open(jsonFile) as freebaseFile:
#         property2region2value = json.loads(freebaseFile.read())
#
#     regions = set([])
#     valueCounter = 0
#     for property, region2value in property2region2value.items():
#         # Check for nan values and remove them
#         for region, value in region2value.items():
#             if not np.isfinite(value):
#                 del region2value[region]
#                 print "REMOVED:", value, " for ", region, " ", property
#         if len(region2value) == 0:
#             del property2region2value[property]
#             print "REMOVED property:", property, " no values left"
#         else:
#             valueCounter += len(region2value)
#             regions = regions.union(set(region2value.keys()))
#
#     print len(property2region2value), " properties"
#     print len(regions), " unique regions"
#     print valueCounter, " values loaded"
#     return property2region2value

def apply_normalizations(costs):
    """Add a 'normalised_matrix' next to each 'cost_matrix' in the values of costs"""

    # TODO - make this have parameters that can be adjusted
    def sigmoid(v):
        return 1 / (1 + np.exp(-v))

    def min_max(lst):
        values = [v for v in lst if v is not None]
        return min(values), max(values)

    def sum_reg(a,lst):
        values = [a*v for v in lst if v is not None]
        return sum(values)

    def sum_exp(a,lst):
        values_exp = []
        for v in lst:
            if v is not None:
                try:
                    values_exp.append(math.exp(a*v))
                except OverflowError:
                    values_exp.append(float(1e10))
        return sum(values_exp)

    def sum_sq(a,lst):
        values_squared = [(a*v)**2 for v in lst if v is not None]
        return sum(values_squared)

    def normalize(v, least, most):
        return 1.0 if least == most else float(v - least) / (most - least)

    def normalize_sums(v, sum):
        return float(v) / sum

    def normalize_dicts_local(lst):
        spans = [min_max(dic.values()) for dic in lst]
        return [{key: normalize(val,*span) for key,val in dic.iteritems()} for dic,span in zip(lst,spans)]

    def normalize_dicts_local_sum(lst):
        sums = [sum_reg(1,dic.values()) for dic in lst]
        return [{key: normalize_sums(val,tempsum) for key,val in dic.iteritems()} for dic,tempsum in zip(lst,sums)]

    def normalize_dicts_local_exp(lst):
        sums_exp = [sum_exp(1,dic.values()) for dic in lst]
        return [{key: normalize_sums(val,tempsum) for key,val in dic.iteritems()} for dic,tempsum in zip(lst,sums_exp)]

    def normalize_dicts_local_sq(lst):
        sums_sq = [sum_sq(1,dic.values()) for dic in lst]
        return [{key: normalize_sums(val,tempsum) for key,val in dic.iteritems()} for dic,tempsum in zip(lst,sums_sq)]

    def normalize_dicts_local_sigmoid(lst):
        return [{key: sigmoid(val) for key,val in dic.iteritems()} for dic in lst]


    for name, value in costs.items():
        # Only normalise the dicts that are not binary
        if int((name.split("_")[-1]))>4:
            value['normalised_matrix'] = normalize_dicts_local(value['cost_matrix'])
            value['normalised_matrix_sum'] = normalize_dicts_local_sum(value['cost_matrix'])
            value['normalised_matrix_sumSquared'] = normalize_dicts_local_sq(value['cost_matrix'])
            value['normalised_matrix_sumExp'] = normalize_dicts_local_exp(value['cost_matrix'])
            value['normalised_matrix_sigmoid'] = normalize_dicts_local_sigmoid(value['cost_matrix'])


def sentence_to_words(sentence, remove_stopwords=False):
    letters_only = re.sub('[^a-zA-Z| LOCATION_SLOT | NUMBER_SLOT]', " ", sentence)
    # [x.strip() for x in re.findall('\s*(\w+|\W+)', line)]
    words = letters_only.lower().split()
    # In Python, searching a set is much faster than searching a list, so convert the stop words to a set
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return (words)


def genCostMatrices(inputSentences, costMatrix, openVector, closedVector, openThresholdVector, closedThresholdVector,
                    costThreshold):
    # This is the MAPE threshold
    global threshold
    for i, sentence in enumerate(inputSentences[:15000]):
        if sentence:

            # Generate all the variables I need
            error = float(sentence['meanAbsError'])
            closedError = float(sentence['closedMeanAbsError'])

            openDict = sentence['openCostDict']
            closedDict = sentence['closedCostDict']

            for key,value in openDict.items():
                if value == inf:
                    openDict[key] = 1e10

            for key,value in closedDict.items():
                if value == inf:
                    openDict[key] = 1e10

            # Now we ensure there is a value for each potential label in the cost vector
            for key, value in openVector.items():
                if key not in openDict:
                    openDict[key] = 1e10

            for key, value in closedVector.items():
                if key not in closedDict:
                    closedDict[key] = 1e10

            openDictThreshold = copy.deepcopy(openDict)
            closedDictThreshold = copy.deepcopy(closedDict)

            # Now we add a threshold cost vector which extends the normal one to add a new label, no_region which is 0 or 1e10inity depending on the results of the MAPE threshold
            openDictThreshold['no_region'] = float(0) if sentence[
                                                             'predictedPropertyOpenThreshold'] == "no_region" else 1e10
            closedDictThreshold['no_region'] = float(0) if sentence[
                                                               'predictedPropertyClosedThreshold'] == "no_region" else 1e10

            extracted = sentence['location-value-pair'].values()[0]
            # No need to account for zero values as not dividing
            if extracted == 0:
                extracted == 0e-10

            # print "Extracted is",extracted

            '''
            Calculate different cost arrays and the best choice
            '''

            openDict_1 = {}
            openDict_2 = {}
            openDict_3 = {}

            closedDict_1 = {}
            closedDict_2 = {}
            closedDict_3 = {}

            for property, value in sentence['closedValues'].items():
                closedDict_1[property] = np.abs(value - extracted)
                closedDict_2[property] = np.abs(value - extracted) / (np.abs(extracted) + np.abs(value))
                closedDict_3[property] = np.abs(value - extracted) / np.abs(extracted + value)

            for property, value in sentence['openValues'].items():
                openDict_1[property] = np.abs(value - extracted)
                openDict_2[property] = np.abs(value - extracted) / (np.abs(extracted) + np.abs(value))
                openDict_3[property] = np.abs(value - extracted) / np.abs(extracted + value)

            for key,value in openDict_1.items():
                if value == inf:
                    openDict_1[key] = 1e10

            for key,value in closedDict_1.items():
                if value == inf:
                    closedDict_1[key] = 1e10

            for key,value in openDict_2.items():
                if value == inf:
                    openDict_2[key] = 1e10

            for key,value in closedDict_2.items():
                if value == inf:
                    closedDict_2[key] = 1e10

            for key,value in openDict_3.items():
                if value == inf:
                    openDict_3[key] = 1e10

            for key,value in closedDict_3.items():
                if value == inf:
                    closedDict_3[key] = 1e10

            # Now we ensure there is a value for each potential label in the cost vectors we have added


            for key, value in openVector.items():
                if key not in openDict_1:
                    openDict_1[key] = 1e10

            for key, value in closedVector.items():
                if key not in closedDict_1:
                    closedDict_1[key] = 1e10

            for key, value in openVector.items():
                if key not in openDict_2:
                    openDict_2[key] = 1e10

            for key, value in closedVector.items():
                if key not in closedDict_2:
                    closedDict_2[key] = 1e10

            for key, value in openVector.items():
                if key not in openDict_3:
                    openDict_3[key] = 1e10

            for key, value in closedVector.items():
                if key not in closedDict_3:
                    closedDict_3[key] = 1e10

            '''
            Get the chosen predictions and their minimum values - this is for the single array case.
            '''

            openPrediction = sentence['predictedPropertyOpen']
            closedPrediction = sentence['predictedPropertyClosed']

            openPrediction_1 = min(openDict_1, key=openDict_1.get)
            closedPrediction_1 = min(closedDict_1, key=closedDict_1.get)
            openAbsError1 = min(openDict_1.itervalues())
            closedAbsError1 = min(closedDict_1.itervalues())

            openPrediction_2 = min(openDict_2, key=openDict_2.get)
            closedPrediction_2 = min(closedDict_2, key=closedDict_2.get)
            openAbsError2 = min(openDict_2.itervalues())
            closedAbsError2 = min(closedDict_2.itervalues())

            openPrediction_3 = min(openDict_3, key=openDict_3.get)
            closedPrediction_3 = min(closedDict_3, key=closedDict_3.get)
            openAbsError3 = min(openDict_3.itervalues())
            closedAbsError3 = min(closedDict_3.itervalues())

            '''
            Get a thresholded version of each cost vector depending on its specific cost method
            '''
            openDictThreshold_1 = copy.deepcopy(openDict_1)
            closedDictThreshold_1 = copy.deepcopy(closedDict_1)
            openDictThreshold_1['no_region'] = float(0) if openAbsError1 > threshold else 1e10
            closedDictThreshold_1['no_region'] = float(0) if closedAbsError1 > threshold else 1e10

            openDictThreshold_2 = copy.deepcopy(openDict_2)
            closedDictThreshold_2 = copy.deepcopy(closedDict_2)
            openDictThreshold_2['no_region'] = float(0) if openAbsError2 > threshold else 1e10
            closedDictThreshold_2['no_region'] = float(0) if closedAbsError2 > threshold else 1e10

            openDictThreshold_3 = copy.deepcopy(openDict_3)
            closedDictThreshold_3 = copy.deepcopy(closedDict_3)
            openDictThreshold_3['no_region'] = float(0) if openAbsError3 > threshold else 1e10
            closedDictThreshold_3['no_region'] = float(0) if closedAbsError3 > threshold else 1e10

            '''
            Calculate the best choice
            '''

            openPredictionThreshold = min(openDictThreshold, key=openDictThreshold.get)
            closedPredictionThreshold = min(closedDictThreshold, key=closedDictThreshold.get)

            openPredictionThreshold_1 = min(openDictThreshold_1, key=openDictThreshold_1.get)
            closedPredictionThreshold_1 = min(closedDictThreshold_1, key=closedDictThreshold_1.get)

            openPredictionThreshold_2 = min(openDictThreshold_2, key=openDictThreshold_2.get)
            closedPredictionThreshold_2 = min(closedDictThreshold_2, key=closedDictThreshold_2.get)

            openPredictionThreshold_3 = min(openDictThreshold_3, key=openDictThreshold_3.get)
            closedPredictionThreshold_3 = min(closedDictThreshold_3, key=closedDictThreshold_3.get)

            '''
            Now we create a layer on the top - cost thresholded versions of the above
            '''



            openDictCostThreshold = {key: (1e10 if value > costThreshold else value) for key, value in
                                     openDict.iteritems()}
            closedDictCostThreshold = {key: (1e10 if value > costThreshold else value) for key, value in
                                       closedDict.iteritems()}

            openDictCostThreshold_1 = {key: (1e10 if value > costThreshold else value) for key, value in
                                       openDict_1.iteritems()}
            closedDictCostThreshold_1 = {key: (1e10 if value > costThreshold else value) for key, value in
                                         closedDict_1.iteritems()}

            openDictCostThreshold_2 = {key: (1e10 if value > costThreshold else value) for key, value in
                                       openDict_2.iteritems()}
            closedDictCostThreshold_2 = {key: (1e10 if value > costThreshold else value) for key, value in
                                         closedDict_2.iteritems()}


            openDictCostThreshold_3 = {key: (1e10 if value > costThreshold else value) for key, value in
                                       openDict_3.iteritems()}
            closedDictCostThreshold_3 = {key: (1e10 if value > costThreshold else value) for key, value in
                                         closedDict_3.iteritems()}

            openDictThresholdCostThreshold = {key: (1e10 if value > costThreshold else value) for key, value in
                                              openDictThreshold.iteritems()}
            closedDictThresholdCostThreshold = {key: (1e10 if value > costThreshold else value) for key, value in
                                                closedDictThreshold.iteritems()}

            openDictThresholdCostThreshold_1 = {key: (1e10 if value > costThreshold else value) for key, value in
                                                openDictThreshold_1.iteritems()}
            closedDictThresholdCostThreshold_1 = {key: (1e10 if value > costThreshold else value) for key, value in
                                                  closedDictThreshold_1.iteritems()}



            openDictThresholdCostThreshold_2 = {key: (1e10 if value > costThreshold else value) for key, value in
                                                openDictThreshold_2.iteritems()}
            closedDictThresholdCostThreshold_2 = {key: (1e10 if value > costThreshold else value) for key, value in
                                                  closedDictThreshold_2.iteritems()}



            openDictThresholdCostThreshold_3 = {key: (1e10 if value > costThreshold else value) for key, value in
                                                openDictThreshold_3.iteritems()}
            closedDictThresholdCostThreshold_3 = {key: (1e10 if value > costThreshold else value) for key, value in
                                                  closedDictThreshold_3.iteritems()}

            '''
            Calculate the best choice
            '''

            openPredictionCostThreshold = min(openDictCostThreshold, key=openDictCostThreshold.get)
            closedPredictionCostThreshold = min(closedDictCostThreshold, key=closedDictCostThreshold.get)

            openPredictionCostThreshold_1 = min(openDictCostThreshold_1, key=openDictCostThreshold_1.get)
            closedPredictionCostThreshold_1 = min(closedDictCostThreshold_1, key=closedDictCostThreshold_1.get)

            openPredictionCostThreshold_2 = min(openDictCostThreshold_2, key=openDictCostThreshold_2.get)
            closedPredictionCostThreshold_2 = min(closedDictCostThreshold_2, key=closedDictCostThreshold_2.get)

            openPredictionCostThreshold_3 = min(openDictCostThreshold_3, key=openDictCostThreshold_3.get)
            closedPredictionCostThreshold_3 = min(closedDictCostThreshold_3, key=closedDictCostThreshold_3.get)

            openPredictionThresholdCostThreshold = min(openDictThresholdCostThreshold,
                                                       key=openDictThresholdCostThreshold.get)
            closedPredictionThresholdCostThreshold = min(closedDictThresholdCostThreshold,
                                                         key=closedDictThresholdCostThreshold.get)

            openPredictionThresholdCostThreshold_1 = min(openDictThresholdCostThreshold_1,
                                                         key=openDictThresholdCostThreshold_1.get)
            closedPredictionThresholdCostThreshold_1 = min(closedDictThresholdCostThreshold_1,
                                                           key=closedDictThresholdCostThreshold_1.get)

            openPredictionThresholdCostThreshold_2 = min(openDictThresholdCostThreshold_2,
                                                         key=openDictThresholdCostThreshold_2.get)
            closedPredictionThresholdCostThreshold_2 = min(closedDictThresholdCostThreshold_2,
                                                           key=closedDictThresholdCostThreshold_2.get)

            openPredictionThresholdCostThreshold_3 = min(openDictThresholdCostThreshold_3,
                                                         key=openDictThresholdCostThreshold_3.get)
            closedPredictionThresholdCostThreshold_3 = min(closedDictThresholdCostThreshold_3,
                                                           key=closedDictThresholdCostThreshold_3.get)

            '''
            Use calculations on the normalised format for the single version, and normalise also
            '''

            for key, data in costMatrix.iteritems():
                # When there is only one cost per training instance
                if str(key).startswith("open"):
                    if str(key).endswith("1"):
                        dict = {key: (0 if key == openPrediction else 1) for key, value in openVector.items()}
                        data['cost_matrix'].append(dict)
                    if str(key).endswith("2"):
                        dict = {key: (0 if key == openPrediction_1 else 1) for key, value in openVector.items()}
                        data['cost_matrix'].append(dict)
                    if str(key).endswith("3"):
                        dict = {key: (0 if key == openPrediction_2 else 1) for key, value in openVector.items()}
                        data['cost_matrix'].append(dict)
                    if str(key).endswith("4"):
                        dict = {key: (0 if key == openPrediction_3 else 1) for key, value in openVector.items()}
                        data['cost_matrix'].append(dict)
                    if str(key).endswith("5"):
                        data['cost_matrix'].append(openDict)
                    if str(key).endswith("6"):
                        data['cost_matrix'].append(openDict_1)
                    if str(key).endswith("7"):
                        data['cost_matrix'].append(openDict_2)
                    if str(key).endswith("8"):
                        data['cost_matrix'].append(openDict_3)
                if str(key).startswith("closed"):
                    if str(key).endswith("1"):
                        dict = {key: (0 if key == closedPrediction else 1) for key, value in closedVector.items()}
                        data['cost_matrix'].append(dict)
                    if str(key).endswith("2"):
                        dict = {key: (0 if key == closedPrediction_1 else 1) for key, value in closedVector.items()}
                        data['cost_matrix'].append(dict)
                    if str(key).endswith("3"):
                        dict = {key: (0 if key == closedPrediction_2 else 1) for key, value in closedVector.items()}
                        data['cost_matrix'].append(dict)
                    if str(key).endswith("4"):
                        dict = {key: (0 if key == closedPrediction_3 else 1) for key, value in closedVector.items()}
                        data['cost_matrix'].append(dict)
                    if str(key).endswith("5"):
                        data['cost_matrix'].append(closedDict)
                    if str(key).endswith("6"):
                        data['cost_matrix'].append(closedDict_1)
                    if str(key).endswith("7"):
                        data['cost_matrix'].append(closedDict_2)
                    if str(key).endswith("8"):
                        data['cost_matrix'].append(closedDict_3)
                if str(key).startswith("threshold"):
                    if str(key).split('_')[1] == "open":
                        if str(key).endswith("1"):
                            dict = {key: (0 if key == openPredictionThreshold else 1) for key, value in openThresholdVector.items()}
                            data['cost_matrix'].append(dict)
                        if str(key).endswith("2"):
                            dict = {key: (0 if key == openPredictionThreshold_1 else 1) for key, value in openThresholdVector.items()}
                            data['cost_matrix'].append(dict)
                        if str(key).endswith("3"):
                            dict = {key: (0 if key == openPredictionThreshold_2 else 1) for key, value in openThresholdVector.items()}
                            data['cost_matrix'].append(dict)
                        if str(key).endswith("4"):
                            dict = {key: (0 if key == openPredictionThreshold_3 else 1) for key, value in openThresholdVector.items()}
                            data['cost_matrix'].append(dict)
                        if str(key).endswith("5"):
                            data['cost_matrix'].append(openDictThreshold)
                        if str(key).endswith("6"):
                            data['cost_matrix'].append(openDictThreshold_1)
                        if str(key).endswith("7"):
                            data['cost_matrix'].append(openDictThreshold_2)
                        if str(key).endswith("8"):
                            data['cost_matrix'].append(openDictThreshold_3)
                    else:
                        if str(key).endswith("1"):
                            dict = {key: (0 if key == closedPredictionThreshold else 1) for key, value in closedThresholdVector.items()}
                            data['cost_matrix'].append(dict)
                        if str(key).endswith("2"):
                            dict = {key: (0 if key == closedPredictionThreshold_1 else 1) for key, value in closedThresholdVector.items()}
                            data['cost_matrix'].append(dict)
                        if str(key).endswith("3"):
                            dict = {key: (0 if key == closedPredictionThreshold_2 else 1) for key, value in closedThresholdVector.items()}
                            data['cost_matrix'].append(dict)
                        if str(key).endswith("4"):
                            dict = {key: (0 if key == closedPredictionThreshold_3 else 1) for key, value in closedThresholdVector.items()}
                            data['cost_matrix'].append(dict)
                        if str(key).endswith("5"):
                            data['cost_matrix'].append(closedDictThreshold)
                        if str(key).endswith("6"):
                            data['cost_matrix'].append(closedDictThreshold_1)
                        if str(key).endswith("7"):
                            data['cost_matrix'].append(closedDictThreshold_2)
                        if str(key).endswith("8"):
                            data['cost_matrix'].append(closedDictThreshold_3)
                if str(key).startswith("costThreshold"):
                    # Not a threshold case
                    if str(key).split('_')[1] == "open":
                        if str(key).endswith("1"):
                            dict = {key: (0 if key == openPredictionCostThreshold else 1) for key, value in openVector.items()}
                            data['cost_matrix'].append(dict)
                        if str(key).endswith("2"):
                            dict = {key: (0 if key == openPredictionCostThreshold_1 else 1) for key, value in openVector.items()}
                            data['cost_matrix'].append(dict)
                        if str(key).endswith("3"):
                            dict = {key: (0 if key == openPredictionCostThreshold_2 else 1) for key, value in openVector.items()}
                            data['cost_matrix'].append(dict)
                        if str(key).endswith("4"):
                            dict = {key: (0 if key == openPredictionCostThreshold_3 else 1) for key, value in openVector.items()}
                            data['cost_matrix'].append(dict)
                        if str(key).endswith("5"):
                            data['cost_matrix'].append(openDictCostThreshold)
                        if str(key).endswith("6"):
                            data['cost_matrix'].append(openDictCostThreshold_1)
                        if str(key).endswith("7"):
                            data['cost_matrix'].append(openDictCostThreshold_2)
                        if str(key).endswith("8"):
                            data['cost_matrix'].append(openDictCostThreshold_3)
                    else:
                        if str(key).endswith("1"):
                            dict = {key: (0 if key == closedPredictionCostThreshold else 1) for key, value in closedVector.items()}
                            data['cost_matrix'].append(dict)
                        if str(key).endswith("2"):
                            dict = {key: (0 if key == closedPredictionCostThreshold_1 else 1) for key, value in closedVector.items()}
                            data['cost_matrix'].append(dict)
                        if str(key).endswith("3"):
                            dict = {key: (0 if key == closedPredictionCostThreshold_2 else 1) for key, value in closedVector.items()}
                            data['cost_matrix'].append(dict)
                        if str(key).endswith("4"):
                            dict = {key: (0 if key == closedPredictionCostThreshold_3 else 1) for key, value in closedVector.items()}
                            data['cost_matrix'].append(dict)
                        if str(key).endswith("5"):
                            data['cost_matrix'].append(closedDictCostThreshold)
                        if str(key).endswith("6"):
                            data['cost_matrix'].append(closedDictCostThreshold_1)
                        if str(key).endswith("7"):
                            data['cost_matrix'].append(closedDictCostThreshold_2)
                        if str(key).endswith("8"):
                            data['cost_matrix'].append(closedDictCostThreshold_3)
                if str(key).startswith("thresholdCostThreshold"):
                    if str(key).split('_')[1] == "open":
                        if str(key).endswith("1"):
                            dict = {key: (0 if key == openPredictionThresholdCostThreshold else 1) for key, value in openThresholdVector.items()}
                            data['cost_matrix'].append(dict)
                        if str(key).endswith("2"):
                            dict = {key: (0 if key == openPredictionThresholdCostThreshold_1 else 1) for key, value in openThresholdVector.items()}
                            data['cost_matrix'].append(dict)
                        if str(key).endswith("3"):
                            dict = {key: (0 if key == openPredictionThresholdCostThreshold_2 else 1) for key, value in openThresholdVector.items()}
                            data['cost_matrix'].append(dict)
                        if str(key).endswith("4"):
                            dict = {key: (0 if key == openPredictionThresholdCostThreshold_3 else 1) for key, value in openThresholdVector.items()}
                            data['cost_matrix'].append(dict)
                        if str(key).endswith("5"):
                            data['cost_matrix'].append(openDictThresholdCostThreshold)
                        if str(key).endswith("6"):
                            data['cost_matrix'].append(openDictThresholdCostThreshold_1)
                        if str(key).endswith("7"):
                            data['cost_matrix'].append(openDictThresholdCostThreshold_2)
                        if str(key).endswith("8"):
                            data['cost_matrix'].append(openDictThresholdCostThreshold_3)
                    else:
                        if str(key).endswith("1"):
                            dict = {key: (0 if key == closedPredictionThresholdCostThreshold else 1) for key, value in closedThresholdVector.items()}
                            data['cost_matrix'].append(dict)
                        if str(key).endswith("2"):
                            dict = {key: (0 if key == closedPredictionThresholdCostThreshold_1 else 1) for key, value in closedThresholdVector.items()}
                            data['cost_matrix'].append(dict)
                        if str(key).endswith("3"):
                            dict = {key: (0 if key == closedPredictionThresholdCostThreshold_2 else 1) for key, value in closedThresholdVector.items()}
                            data['cost_matrix'].append(dict)
                        if str(key).endswith("4"):
                            dict = {key: (0 if key == closedPredictionThresholdCostThreshold_3 else 1) for key, value in closedThresholdVector.items()}
                            data['cost_matrix'].append(dict)
                        if str(key).endswith("5"):
                            data['cost_matrix'].append(closedDictThresholdCostThreshold)
                        if str(key).endswith("6"):
                            data['cost_matrix'].append(closedDictThresholdCostThreshold_1)
                        if str(key).endswith("7"):
                            data['cost_matrix'].append(closedDictThresholdCostThreshold_2)
                        if str(key).endswith("8"):
                            data['cost_matrix'].append(closedDictThresholdCostThreshold_3)

# for key, value in costMatrix.iteritems():
#     print key," array is ", value['cost_matrix'],"\n"

def addFeatures(string,featureStrings):
    string += "| " + featureStrings
    return string


def generateDatFiles(fullCostDict, trainingLabels, closedTrainingLabels, trainingFeatures, bigramFeatures, depgramFeatures,
                     wordgramFeatures):
    # TODO - make sure this opens a fresh file all the time and replaces

    print "There are this many unique labels", len(set(trainingLabels))
    print "There are this many closed unique labels", len(set(closedTrainingLabels))

    print "Generating VW files...\n"

    for model, dict in fullCostDict.items():
        arow_file = open(dict['words_arow_path'], 'w')
        arow_file_bigrams = open(dict['bigrams_arow_path'], 'w')
        arow_file_depgrams = open(dict['depgrams_arow_path'], 'w')
        arow_file_wordgrams = open(dict['wordgrams_arow_path'], 'w')

        # Now open all the normal versions

        arow_fileN = open(dict['words_arow_path_normal'],'w')
        arow_fileN1 =  open(dict['words_arow_path_normal1'],'w')
        arow_fileN2 =  open(dict['words_arow_path_normal2'],'w')
        arow_fileN3 =  open(dict['words_arow_path_normal3'],'w')
        arow_fileN4 =  open(dict['words_arow_path_normal4'],'w')

        arow_file_bigramsN = open(dict['bigrams_arow_path_normal'],'w')
        arow_file_bigramsN1 = open(dict['bigrams_arow_path_normal1'],'w')
        arow_file_bigramsN2 = open(dict['bigrams_arow_path_normal2'],'w')
        arow_file_bigramsN3 = open(dict['bigrams_arow_path_normal3'],'w')
        arow_file_bigramsN4 = open(dict['bigrams_arow_path_normal4'],'w')

        arow_file_depgramsN = open(dict['depgrams_arow_path_normal'],'w')
        arow_file_depgramsN1 = open(dict['depgrams_arow_path_normal1'],'w')
        arow_file_depgramsN2 = open(dict['depgrams_arow_path_normal2'],'w')
        arow_file_depgramsN3 = open(dict['depgrams_arow_path_normal3'],'w')
        arow_file_depgramsN4 = open(dict['depgrams_arow_path_normal4'],'w')

        arow_file_wordgramsN = open(dict['wordgrams_arow_path_normal'],'w')
        arow_file_wordgramsN1 = open(dict['wordgrams_arow_path_normal1'],'w')
        arow_file_wordgramsN2 = open(dict['wordgrams_arow_path_normal2'],'w')
        arow_file_wordgramsN3 = open(dict['wordgrams_arow_path_normal3'],'w')
        arow_file_wordgramsN4 = open(dict['wordgrams_arow_path_normal4'],'w')

        if 'cost_matrix' in dict:
            for i, (label, closedLabel, costDict, features, bigrams, depgrams, worgrams) in enumerate(
                zip(trainingLabels, closedTrainingLabels, dict['cost_matrix'],trainingFeatures, bigramFeatures,
                    depgramFeatures, wordgramFeatures)):
                depgrams = depgrams.decode('utf-8')
                arow_line = ""
                arow_line_bigrams = ""
                arow_line_depgrams = ""
                arow_line_wordgrams = ""
                for count, (lb, cost) in enumerate(costDict.items()):
                    arow_line += str(lb) + ":" + str(cost) + " "
                    arow_line_bigrams += str(lb) + ":" + str(cost) + " "
                    arow_line_depgrams += str(lb) + ":" + str(cost) + " "
                    arow_line_wordgrams += str(lb) + ":" + str(cost) + " "

                arowLines = [arow_line]
                arowLines = map(lambda x: x + "| " + features, arowLines)
                arowLinesBigrams = [arow_line_bigrams]
                arowLinesBigrams = map(lambda x: x + "| " + bigrams, arowLinesBigrams)
                arowLinesDepgrams = [arow_line_depgrams]
                arowLinesDepgrams = map(lambda x: x + "| " + depgrams.encode('utf-8'), arowLinesDepgrams)
                arowLinesWordgrams = [arow_line_wordgrams]
                arowLinesWordgrams = map(lambda x: x + "| " + worgrams.encode('utf-8'), arowLinesWordgrams)
                arowLines = arowLines + arowLinesBigrams + arowLinesDepgrams + arowLinesWordgrams
                arow_files = [arow_file, arow_file_bigrams,arow_file_depgrams,arow_file_wordgrams]
                for line, file in zip(arowLines,arow_files):
                    file.write(line + "\n")
        # How to only do this if normalised matrix exists
        if 'normalised_matrix' in dict:
            for i, (label, closedLabel, normalCostDict, normalCostDict1, normalCostDict2,normalCostDict3,normalCostDict4,features, bigrams, depgrams, worgrams) in enumerate(
                    zip(trainingLabels, closedTrainingLabels, dict['normalised_matrix'],dict['normalised_matrix_sum'],dict['normalised_matrix_sumSquared'],dict['normalised_matrix_sumExp'],dict['normalised_matrix_sigmoid'],trainingFeatures, bigramFeatures,
                        depgramFeatures, wordgramFeatures)):

                depgrams = depgrams.decode('utf-8')
                arow_lineN = ""
                arow_line_bigramsN = ""
                arow_line_depgramsN = ""
                arow_line_wordgramsN = ""

                arow_lineN1 = ""
                arow_line_bigramsN1 = ""
                arow_line_depgramsN1 = ""
                arow_line_wordgramsN1 = ""

                arow_lineN2 = ""
                arow_line_bigramsN2 = ""
                arow_line_depgramsN2 = ""
                arow_line_wordgramsN2 = ""

                arow_lineN3 = ""
                arow_line_bigramsN3 = ""
                arow_line_depgramsN3 = ""
                arow_line_wordgramsN3 = ""

                arow_lineN4 = ""
                arow_line_bigramsN4 = ""
                arow_line_depgramsN4 = ""
                arow_line_wordgramsN4 = ""

                for count, ((lb1, cost1),(lb2, cost2),(lb3, cost3),(lb4, cost4),(lb5, cost5)) in enumerate(zip(normalCostDict.items(),normalCostDict1.items(),normalCostDict2.items(),normalCostDict3.items(),normalCostDict4.items())):

                    arow_lineN += str(lb1) + ":" + str(cost1) + " "
                    arow_line_bigramsN += str(lb1) + ":" + str(cost1) + " "
                    arow_line_depgramsN += str(lb1) + ":" + str(cost1) + " "
                    arow_line_wordgramsN += str(lb1) + ":" + str(cost1) + " "

                    arow_lineN1 += str(lb2) + ":" + str(cost2) + " "
                    arow_line_bigramsN1 += str(lb2) + ":" + str(cost2) + " "
                    arow_line_depgramsN1 += str(lb2) + ":" + str(cost2) + " "
                    arow_line_wordgramsN1 += str(lb2) + ":" + str(cost2) + " "

                    arow_lineN2 += str(lb3) + ":" + str(cost3) + " "
                    arow_line_bigramsN2 += str(lb3) + ":" + str(cost3) + " "
                    arow_line_depgramsN2 += str(lb3) + ":" + str(cost3) + " "
                    arow_line_wordgramsN2 += str(lb3) + ":" + str(cost3) + " "

                    arow_lineN3 += str(lb4) + ":" + str(cost4) + " "
                    arow_line_bigramsN3 += str(lb4) + ":" + str(cost4) + " "
                    arow_line_depgramsN3 += str(lb4) + ":" + str(cost4) + " "
                    arow_line_wordgramsN3 += str(lb4) + ":" + str(cost4) + " "

                    arow_lineN4 += str(lb5) + ":" + str(cost5) + " "
                    arow_line_bigramsN4 += str(lb5) + ":" + str(cost5) + " "
                    arow_line_depgramsN4 += str(lb5) + ":" + str(cost5) + " "
                    arow_line_wordgramsN4 += str(lb5) + ":" + str(cost5) + " "

                arowLines = [arow_lineN,arow_lineN1,arow_lineN2,arow_lineN3,arow_lineN4]
                arowLines = map(lambda x: x + "| " + features, arowLines)

                arowLinesBigrams = [arow_line_bigramsN,arow_line_bigramsN1,arow_line_bigramsN2,arow_line_bigramsN3,arow_line_bigramsN4]
                arowLinesBigrams = map(lambda x: x + "| " + bigrams, arowLinesBigrams)

                arowLinesDepgrams = [arow_line_depgramsN,arow_line_depgramsN1,arow_line_depgramsN2,arow_line_depgramsN3,arow_line_depgramsN4]
                arowLinesDepgrams = map(lambda x: x + "| " + depgrams.encode('utf-8'), arowLinesDepgrams)


                arowLinesWordgrams = [arow_line_wordgramsN,arow_line_wordgramsN1,arow_line_wordgramsN2,arow_line_wordgramsN3,arow_line_wordgramsN4]
                arowLinesWordgrams = map(lambda x: x + "| " + worgrams.encode('utf-8'), arowLinesWordgrams)

                arowLines = arowLines + arowLinesBigrams + arowLinesDepgrams + arowLinesWordgrams

                arow_files = [arow_fileN,arow_fileN1, arow_fileN2,arow_fileN3,arow_fileN4, arow_file_bigramsN,arow_file_bigramsN1,arow_file_bigramsN2,arow_file_bigramsN3,arow_file_bigramsN4,arow_file_depgramsN,arow_file_depgramsN1,arow_file_depgramsN2,arow_file_depgramsN3,arow_file_depgramsN4,arow_file_wordgramsN,arow_file_wordgramsN1,arow_file_wordgramsN2,arow_file_wordgramsN3,arow_file_wordgramsN4]

                for line, file in zip(arowLines,arow_files):
                    file.write(line + "\n")

def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])


def training_features(inputSentences):
    global vectorizer
    # global train_wordbigram_list_sparse
    # global train_bigram_list_sparse
    # global train_wordlist_sparse
    # global train_gramlist_sparse
    print "Preparing the raw features...\n"
    for i, sentence in enumerate(inputSentences[:15000]):
        # Dont train if the sentence contains a random region we don't care about
        # and sentence['predictedRegion'] in properties
        if sentence:
            # print "Sentence is",sentence
            # Regardless of anything else, an open evaluation includes all sentences
            words = (" ").join(sentence_to_words(sentence['parsedSentence'], True))
            word_list = sentence_to_words(sentence['parsedSentence'], True)
            # print "words are", words
            bigrams = ""
            if "depPath" in sentence.keys():
                bigrams = [("+").join(bigram).encode('utf-8') for bigram in sentence['depPath']]
                bigrams = (' ').join(map(str, bigrams))
                # print "Bigrams are",bigrams
                # print "Wordgrams are",words+bigrams.decode("utf-8")
            bigrams = ('').join(bigrams)
            # print "Bigrams are",bigrams
            train_bigram_list.append(bigrams)

            train_wordbigram_list.append(words + " " + bigrams.decode("utf-8"))
            train_wordlist.append(words)

            wordgrams = find_ngrams(word_list, 2)
            for i, grams in enumerate(wordgrams):
                wordgrams[i] = '+'.join(grams)
            wordgrams = (" ").join(wordgrams)
            train_gramlist.append(wordgrams)

            train_property_labels.append(sentence['predictedPropertyOpen'])
            train_property_labels_threshold.append(sentence['predictedPropertyOpenThreshold'])
            closed_train_property_labels.append(sentence['predictedPropertyClosed'])
            closed_train_property_labels_threshold.append(sentence['predictedPropertyClosedThreshold'])

    # print "These are the clean words in the training sentences: ", train_wordlist
    # print "These are the labels in the training sentences: ", train_labels
    print "Creating the bag of words...\n"
    # train_wordlist_sparse = vectorizer.fit_transform(train_wordlist).toarray().astype(np.float)
    # train_gramlist_sparse = vectorizer.fit_transform(train_gramlist).toarray().astype(np.float)
    # train_bigram_list_sparse = vectorizer.fit_transform(train_bigram_list).toarray().astype(np.float)
    # train_wordbigram_list_sparse = vectorizer.fit_transform(train_wordbigram_list).toarray().astype(np.float)


def test_features(testSentences):
    global vectorizer
    # global test_gramlist_sparse
    # global test_bigram_list_sparse
    # global test_wordbigram_list_sparse
    # global test_wordlist_sparse
    # TODO - ensure that even the multinomial was calculated using non-TFID features
    print "Preparing the raw test features...\n"
    for sentence in testSentences:
        if sentence['parsedSentence'] != {} and sentence['mape_label'] != {}:
            words = " ".join(sentence_to_words(sentence['parsedSentence'], True))
            word_list = sentence_to_words(sentence['parsedSentence'], True)
            wordgrams = find_ngrams(word_list, 2)
            for i, grams in enumerate(wordgrams):
                wordgrams[i] = '+'.join(grams)
            wordgrams = (" ").join(wordgrams)
            test_gramlist.append(wordgrams)
            test_wordlist.append(words)
            test_property_labels.append(sentence['property'])
            bigrams = sentence['depPath']
            # print "Test Bigrams are",bigrams
            test_bigram_list.append(bigrams)
            test_wordbigram_list.append(words + " " + bigrams.decode("utf-8"))
    # print "These are the clean words in the test sentences: ", clean_test_sentences
    # print "These are the mape labels in the test sentences: ", binary_test_labels
    # test_wordlist_sparse = vectorizer.fit_transform(test_wordlist).toarray().astype(np.float)
    # test_wordbigram_list_sparse = vectorizer.fit_transform(test_wordbigram_list).toarray().astype(np.float)
    # test_bigram_list_sparse = vectorizer.fit_transform(test_bigram_list).toarray().astype(np.float)
    # test_gramlist_sparse = vectorizer.fit_transform(test_gramlist).toarray().astype(np.float)

    # test_data_features = vectorizer.transform(clean_test_sentences)
    # test_data_features = test_data_features.toarray()
    # test_data_features = test_data_features.astype(np.float)


if __name__ == "__main__":
    # training data
    # load the sentence file for training

    # property2region2value = loadMatrix(sys.argv[1])

    with open(sys.argv[2]) as trainingSentences:
        pattern2regions = json.loads(trainingSentences.read())

    print "We have ", len(pattern2regions), " training sentences."
    # We load in the allowable features and also no_region

    with open(sys.argv[3]) as testSentences:
        testSentences = json.loads(testSentences.read())

    threshold = testSentences[0]['threshold']

    print "APE Threshold for no region label is ", threshold

    # This is how we threshold across costs
    costThreshold = float(sys.argv[8])

    print "Cost Threshold is ", costThreshold

    with open(sys.argv[4]) as featuresKept:
        properties = json.loads(featuresKept.read())
    # properties.append("no_region")
    # print "Properties are ", properties

    print "Length of final sentences is", len(testSentences)

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

    # train_wordlist_sparse = []
    # test_wordlist_sparse = []
    #
    # train_gramlist_sparse = []
    # test_gramlist_sparse = []
    #
    # train_bigram_list_sparse = []
    # test_bigram_list_sparse = []
    #
    # train_wordbigram_list_sparse = []
    # test_wordbigram_list_sparse = []

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

    vectorizer = Pipeline([('vect', CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None,
                                                    max_features=5000)),
                           ('tfidf', TfidfTransformer(use_idf=True, norm='l2', sublinear_tf=True))])

    print "Getting all the words in the training sentences...\n"

    # This both sorts out the features and the training labels
    training_features(pattern2regions)

    print len(train_wordlist), "sets of training features"

    trainingLabels = len(pattern2regions)
    positiveOpenTrainingLabels = len(train_property_labels_threshold) - train_property_labels_threshold.count(
        "no_region")
    negativeOpenTrainingLabels = train_property_labels_threshold.count("no_region")
    positiveClosedTrainingLabels = len(
        closed_train_property_labels_threshold) - closed_train_property_labels_threshold.count(
        "no_region")
    negativeClosedTrainingLabels = closed_train_property_labels_threshold.count(
        "no_region")

    # Fit the logistic classifiers to the training set, using the bag of words as features

    openTrainingClasses = len(set(train_property_labels))
    openTrainingClassesThreshold = len(set(train_property_labels_threshold))
    closedTrainingClasses = len(set(closed_train_property_labels))
    closedTrainingClassesThreshold = len(set(closed_train_property_labels_threshold))

    print "Get a bag of words for the test set, and convert to a numpy array\n"

    test_features(testSentences)

    print len(test_wordlist), "sets of test features"

    print "There are ", len(set(train_property_labels)), "open training classes"
    print "There are ", len(set(train_property_labels_threshold)), "open training classes w/threshold"
    print "There are ", len(set(closed_train_property_labels)), "closed training properties"
    print "There are ", len(set(closed_train_property_labels_threshold)), "closed training properties w/ threshold"

    '''
    Cost sensitive classification
    '''
    cost_matrices = {}

    for x in range(1, 8):
        cost_matrices["open_cost_{0}".format(x)] = {'cost_matrix': []}
        cost_matrices["closed_cost_{0}".format(x)] = {'cost_matrix': []}
        cost_matrices["threshold_open_cost_{0}".format(x)] = {'cost_matrix': []}
        cost_matrices["threshold_closed_cost_{0}".format(x)] = {'cost_matrix': []}
        cost_matrices["costThreshold_open_cost_{0}".format(x)] = {'cost_matrix': []}
        cost_matrices["costThreshold_closed_cost_{0}".format(x)] = {'cost_matrix': []}
        cost_matrices["thresholdCostThreshold_open_cost_{0}".format(x)] = {'cost_matrix': []}
        cost_matrices["thresholdCostThreshold_closed_cost_{0}".format(x)] = {'cost_matrix': []}
        # cost_matrices["normalised_open_cost_{0}".format(x)]=[]
        # cost_matrices["normalised_closed_cost_{0}".format(x)]=[]

    model_path = os.path.join(sys.argv[6] + '/models.txt')

    with open(model_path, "wb") as out:
        json.dump(cost_matrices, out, indent=4)

    openKeySet = set(train_property_labels)
    openMapping = {val: key + 1 for key, val in enumerate(list(openKeySet))}
    openInvMapping = {key + 1: val for key, val in enumerate(list(openKeySet))}
    open_mapping_path = os.path.join(sys.argv[6] + '/open_label_mapping.txt')
    with open(open_mapping_path, "wb") as out:
        json.dump(openInvMapping, out, indent=4)

    openKeySetThreshold = copy.deepcopy(openKeySet)
    openKeySetThreshold |= {"no_region"}
    openMappingThreshold = {val: key + 1 for key, val in enumerate(list(openKeySetThreshold))}
    openInvMappingThreshold = {key + 1: val for key, val in enumerate(list(openKeySetThreshold))}
    open_mapping_path_threshold = os.path.join(sys.argv[6] + '/open_label_mapping_threshold.txt')
    with open(open_mapping_path_threshold, "wb") as out:
        json.dump(openInvMappingThreshold, out, indent=4)

    closedKeySet = set(closed_train_property_labels)
    closedMapping = {val: key + 1 for key, val in enumerate(list(closedKeySet))}
    closedInvMapping = {key + 1: val for key, val in enumerate(list(closedKeySet))}
    closed_mapping_path = os.path.join(sys.argv[6] + '/closed_label_mapping.txt')
    with open(closed_mapping_path, "wb") as out:
        json.dump(closedInvMapping, out, indent=4)

    closedKeySetThreshold = copy.deepcopy(closedKeySet)
    closedKeySetThreshold |= {"no_region"}
    closedMappingThreshold = {val: key + 1 for key, val in enumerate(list(closedKeySetThreshold))}
    closedInvMappingThreshold = {key + 1: val for key, val in enumerate(list(closedKeySetThreshold))}
    closed_mapping_path_threshold = os.path.join(sys.argv[6] + '/closed_label_mapping_threshold.txt')
    with open(closed_mapping_path_threshold, "wb") as out:
        json.dump(closedInvMappingThreshold, out, indent=4)

    print "Generating cost matrices...\n"
    genCostMatrices(pattern2regions, cost_matrices, openMapping, closedMapping, openMappingThreshold,
                    closedMappingThreshold, costThreshold)

    for i, value in enumerate(train_property_labels):
        train_property_labels[i] = openMapping[value]

    for i, value in enumerate(closed_train_property_labels):
        closed_train_property_labels[i] = closedMapping[value]


    '''
    Convert the labels to number not text format.
    '''
    final_cost_matrices = {}

    for x in range(1, 8):
        final_cost_matrices["open_cost_{0}".format(x)] = {'cost_matrix': []}
        final_cost_matrices["closed_cost_{0}".format(x)] = {'cost_matrix': []}
        final_cost_matrices["threshold_open_cost_{0}".format(x)] = {'cost_matrix': []}
        final_cost_matrices["threshold_closed_cost_{0}".format(x)] = {'cost_matrix': []}
        final_cost_matrices["costThreshold_open_cost_{0}".format(x)] = {'cost_matrix': []}
        final_cost_matrices["costThreshold_closed_cost_{0}".format(x)] = {'cost_matrix': []}
        final_cost_matrices["thresholdCostThreshold_open_cost_{0}".format(x)] = {'cost_matrix': []}
        final_cost_matrices["thresholdCostThreshold_closed_cost_{0}".format(x)] = {'cost_matrix': []}

    # These are all the file paths for all the models
    for model,dict in final_cost_matrices.items():
        # dict['words_path'] = os.path.join(sys.argv[5] + model + "_words.dat")
        dict['words_arow_path'] = os.path.join(sys.argv[6] + model + "_"+ str(threshold)+"_"+str(costThreshold)+"_words.dat")
        dict['words_arow_path_normal'] = os.path.join(sys.argv[6]+model+"_"+ str(threshold)+"_"+str(costThreshold)+"_normal_words.dat")
        dict['words_arow_path_normal1'] = os.path.join(sys.argv[6]+model+"_"+ str(threshold)+"_"+str(costThreshold)+"_normal1_words.dat")
        dict['words_arow_path_normal2'] = os.path.join(sys.argv[6]+model+"_"+ str(threshold)+"_"+str(costThreshold)+"_normal2_words.dat")
        dict['words_arow_path_normal3'] = os.path.join(sys.argv[6]+model+"_"+ str(threshold)+"_"+str(costThreshold)+"_normal3_words.dat")
        dict['words_arow_path_normal4'] = os.path.join(sys.argv[6]+model+"_"+ str(threshold)+"_"+str(costThreshold)+"_normal4_words.dat")

        # dict['bigrams_path'] = os.path.join(sys.argv[5]+model+"_bigrams.dat")
        dict['bigrams_arow_path'] = os.path.join(sys.argv[6]+model+"_"+ str(threshold)+"_"+str(costThreshold)+"_bigrams.dat")
        dict['bigrams_arow_path_normal'] = os.path.join(sys.argv[6]+model+"_"+ str(threshold)+"_"+str(costThreshold)+"_normal_bigrams.dat")
        dict['bigrams_arow_path_normal1'] = os.path.join(sys.argv[6]+model+"_"+ str(threshold)+"_"+str(costThreshold)+"_normal1_bigrams.dat")
        dict['bigrams_arow_path_normal2'] = os.path.join(sys.argv[6]+model+"_"+ str(threshold)+"_"+str(costThreshold)+"_normal2_bigrams.dat")
        dict['bigrams_arow_path_normal3'] = os.path.join(sys.argv[6]+model+"_"+ str(threshold)+"_"+str(costThreshold)+"_normal3_bigrams.dat")
        dict['bigrams_arow_path_normal4'] = os.path.join(sys.argv[6]+model+"_"+ str(threshold)+"_"+str(costThreshold)+"_normal4_bigrams.dat")

        # dict['depgrams_path'] = os.path.join(sys.argv[5]+model+"_depgrams.dat")
        dict['depgrams_arow_path'] = os.path.join(sys.argv[6]+model+"_"+ str(threshold)+"_"+str(costThreshold)+"_depgrams.dat")
        dict['depgrams_arow_path_normal'] = os.path.join(sys.argv[6]+model+"_"+ str(threshold)+"_"+str(costThreshold)+"_normal_depgrams.dat")
        dict['depgrams_arow_path_normal1'] = os.path.join(sys.argv[6]+model+"_"+ str(threshold)+"_"+str(costThreshold)+"_normal1_depgrams.dat")
        dict['depgrams_arow_path_normal2'] = os.path.join(sys.argv[6]+model+"_"+ str(threshold)+"_"+str(costThreshold)+"_normal2_depgrams.dat")
        dict['depgrams_arow_path_normal3'] = os.path.join(sys.argv[6]+model+"_"+ str(threshold)+"_"+str(costThreshold)+"_normal3_depgrams.dat")
        dict['depgrams_arow_path_normal4'] = os.path.join(sys.argv[6]+model+"_"+ str(threshold)+"_"+str(costThreshold)+"_normal4_depgrams.dat")
        # dict['wordgrams_path'] = os.path.join(sys.argv[5]+model+"_wordgrams.dat")
        dict['wordgrams_arow_path'] = os.path.join(sys.argv[6]+model+"_"+ str(threshold)+"_"+str(costThreshold)+"_wordgrams.dat")
        dict['wordgrams_arow_path_normal'] = os.path.join(sys.argv[6]+model+"_"+ str(threshold)+"_"+str(costThreshold)+"_normal_wordgrams.dat")
        dict['wordgrams_arow_path_normal1'] = os.path.join(sys.argv[6]+model+"_"+ str(threshold)+"_"+str(costThreshold)+"_normal1_wordgrams.dat")
        dict['wordgrams_arow_path_normal2'] = os.path.join(sys.argv[6]+model+"_"+ str(threshold)+"_"+str(costThreshold)+"_normal2_wordgrams.dat")
        dict['wordgrams_arow_path_normal3'] = os.path.join(sys.argv[6]+model+"_"+ str(threshold)+"_"+str(costThreshold)+"_normal3_wordgrams.dat")
        dict['wordgrams_arow_path_normal4'] = os.path.join(sys.argv[6]+model+"_"+ str(threshold)+"_"+str(costThreshold)+"_normal4_wordgrams.dat")

    for (model, arr),(finalModel, finalDict) in zip(cost_matrices.items(),final_cost_matrices.items()):
        if model.startswith("threshold") or model.startswith("thresholdCostThreshold"):
            if str(model).split('_')[1] == "open":
                for dict in arr['cost_matrix']:
                    tempDict = {}
                    for key,value in dict.items():
                        tempDict[openMappingThreshold[key]] = value
                        # print "tempDict",tempDict
                    finalDict['cost_matrix'].append(tempDict)
                        # dict[openMappingThreshold[key]] = dict.pop(key)
            else:
                for dict in arr['cost_matrix']:
                    tempDict = {}
                    for key,value in dict.items():
                        tempDict[closedMappingThreshold[key]] = value
                        # print "tempDict",tempDict
                    finalDict['cost_matrix'].append(tempDict)
                        # dict[closedMappingThreshold[key]] = dict.pop(key)
        else:
            if str(model).split('_')[1]=="open" or str(model).split('_')[0] == "open":
                for dict in arr['cost_matrix']:
                    tempDict = {}
                    for key,value in dict.items():
                        tempDict[openMapping[key]] = value
                        # print "tempDict",tempDict
                    finalDict['cost_matrix'].append(tempDict)
                        # dict[openMapping[key]] = dict.pop(key)
            else:
                for dict in arr['cost_matrix']:
                    tempDict = {}
                    for key,value in dict.items():
                        tempDict[closedMapping[key]] = value
                        # print "tempDict",tempDict
                    finalDict['cost_matrix'].append(tempDict)
                        # print "Closed"
                        # print "model",model
                        # dict[closedMapping[key]] = dict.pop(key)

            # print "Post-dict",dict

    print "Generating normalised versions of cost matrices...\n"

    apply_normalizations(final_cost_matrices)

    # This is to output the final predictions

    model_path = os.path.join(sys.argv[6] + '/cost_matrices_final.txt')

    with open(model_path, "wb") as out:
        json.dump(final_cost_matrices, out, indent=4)


    '''
    Generate different training files
    '''

    print "Generating training files...\n"
    generateDatFiles(final_cost_matrices, train_property_labels, closed_train_property_labels, train_wordlist, train_gramlist,
                     train_bigram_list, train_wordbigram_list)

    '''
    Generate different versions of the test files.
    '''
    print "Generating test files...\n"
    # Generate test files
    arow_testfile = open(os.path.join(sys.argv[6] + "words_test.test"), 'w')
    # vowpal_testfile = open(os.path.join(sys.argv[5] + "words_test.dat"), 'w')

    arow_testfile_bigrams = open(os.path.join(sys.argv[6] + "bigrams_test.test"), 'w')
    # vowpal_testfile_bigrams = open(os.path.join(sys.argv[5] + "bigrams_test.dat"), 'w')

    arow_testfile_depgrams = open(os.path.join(sys.argv[6] + "depgrams_test.test"), 'w')
    # vowpal_testfile_depgrams = open(os.path.join(sys.argv[5] + "depgrams_test.dat"), 'w')

    arow_testfile_wordgrams = open(os.path.join(sys.argv[6] + "wordgrams_test.test"), 'w')
    # vowpal_testfile_wordgrams = open(os.path.join(sys.argv[5] + "wordgrams_test.dat"), 'w')

    for i, (features, bigrams, depgrams, wordgrams) in enumerate(
            zip(test_wordlist, test_gramlist, test_bigram_list, test_wordbigram_list)):
        # vowpal_line = str(i) + "| " + features
        arow_line = features

        # vowpal_line_bigrams = str(i) + "| " + bigrams.encode('utf-8')
        arow_line_bigrams = bigrams.encode('utf-8')

        # vowpal_line_depgrams = str(i) + "| " + depgrams.encode('utf-8')
        arow_line_depgrams = depgrams.encode('utf-8')

        # vowpal_line_wordgrams = str(i) + "| " + wordgrams.encode('utf-8')
        arow_line_wordgrams = wordgrams.encode('utf-8')

        # Now we write the test files
        # vowpal_testfile.write(vowpal_line + "\n")
        arow_testfile.write(arow_line + "\n")

        # vowpal_testfile_bigrams.write(vowpal_line_bigrams + "\n")
        arow_testfile_bigrams.write(arow_line_bigrams + "\n")

        # vowpal_testfile_depgrams.write(vowpal_line_depgrams + "\n")
        arow_testfile_depgrams.write(arow_line_depgrams + "\n")

        # vowpal_testfile_wordgrams.write(vowpal_line_wordgrams + "\n")
        arow_testfile_wordgrams.write(arow_line_wordgrams + "\n")

