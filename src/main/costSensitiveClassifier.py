import math
import re
from nltk.corpus import stopwords  # Import the stop word list
import json
import numpy as np
import pandas as pd
import sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import os
from sklearn.pipeline import Pipeline
import scipy.stats as st
from collections import defaultdict
import statsmodels.robust.scale.mad as meanabsdev

xs = defaultdict(list)
from numpy import inf
import operator

# This is about loading any file with property: region:value format
def loadMatrix(jsonFile):
    print "loading from file " + jsonFile
    with open(jsonFile) as freebaseFile:
        property2region2value = json.loads(freebaseFile.read())

    regions = set([])
    valueCounter = 0
    for property, region2value in property2region2value.items():
        # Check for nan values and remove them
        for region, value in region2value.items():
            if not np.isfinite(value):
                del region2value[region]
                print "REMOVED:", value, " for ", region, " ", property
        if len(region2value) == 0:
            del property2region2value[property]
            print "REMOVED property:", property, " no values left"
        else:
            valueCounter += len(region2value)
            regions = regions.union(set(region2value.keys()))

    print len(property2region2value), " properties"
    print len(regions),  " unique regions"
    print valueCounter, " values loaded"
    return property2region2value


def sentence_to_words(sentence, remove_stopwords=False):
    letters_only = re.sub('[^a-zA-Z| LOCATION_SLOT | NUMBER_SLOT]', " ", sentence)
    # [x.strip() for x in re.findall('\s*(\w+|\W+)', line)]
    words = letters_only.lower().split()
    # In Python, searching a set is much faster than searching a list, so convert the stop words to a set
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return (words)

def genCostMatrices(inputSentences, costMatrix,openVector, closedVector):
    global property2region2value

    freebaseDF = pd.DataFrame(property2region2value)

    property2rangeMin = freebaseDF.apply(np.min,axis=1).to_dict()
    property2rangeStd = freebaseDF.apply(np.std,axis=1).to_dict()
    property2rangeMean = freebaseDF.apply(np.mean,axis=1).to_dict()
    property2rangeMax = freebaseDF.apply(np.max,axis=1).to_dict()
    # Now we use the spread of values across properties in the KB
    property2range = {key: property2rangeMax[key] - property2rangeMin.get(key, 0) for key in property2rangeMax.keys()}

    # Now we use the spread of values across the sentences in the training data
    stdValue = np.std([sentence['location-value-pair'].values()[0]for i,sentence in enumerate((inputSentences[:15000]))])
    minValue = np.min([sentence['location-value-pair'].values()[0]for i,sentence in enumerate((inputSentences[:15000]))])
    maxValue = np.max([sentence['location-value-pair'].values()[0]for i,sentence in enumerate((inputSentences[:15000]))])
    meanValue = np.mean([sentence['location-value-pair'].values()[0]for i,sentence in enumerate((inputSentences[:15000]))])
    range = maxValue - minValue

    for i, sentence in enumerate(inputSentences[:15000]):
        if sentence:

            # Generate all the variables I need
            error = float(sentence['meanAbsError'])
            closedError = float(sentence['closedMeanAbsError'])

            openDict = sentence['openCostDict']
            closedDict = sentence['closedCostDict']

            extracted = sentence['location-value-pair'].values()[0]
            if extracted == 0:
                extracted == 0e-10

            '''
            Get the normalised version of the value globally
            '''

            extractedNormal = (extracted-minValue)/range
            # print "extractedNormal",extractedNormal
            #
            extractedNormalZScore = np.abs((extracted-meanValue)/stdValue)
            #
            # print "extractedNormal",extractedNormal

            openAPE = np.array(sentence['openCostArr'])
            closedAPE = np.array(sentence['closedCostArr'])

            # print "Open APE is",openAPE

            openAPE[openAPE == inf] = 1e10
            closedAPE[closedAPE == inf] = 1e10

            # for i,value in enumerate(np.array(openAPE)):
            #     if i==inf:
            #         openAPE[i] = 1e10
            #
            # for i,value in enumerate(np.array(closedAPE)):
            #     if i==inf:
            #         closedAPE[i] = 1e10

            # print "Open APE is",openAPE

            '''
            Calculate different cost arrays and the best choice
            '''

            openAPE_1 = []
            openAPE_2 = []
            openAPE_3 = []

            closedAPE_1 = []
            closedAPE_2 = []
            closedAPE_3 = []

            openDict_1 = {}
            openDict_2 = {}
            openDict_3 = {}

            closedDict_1 = {}
            closedDict_2 = {}
            closedDict_3 = {}

            # Now everything is prenormalised values

            normalisedClosedValues = {}
            normalisedOpenValues = {}

            normalisedClosedValues = {}
            normalisedOpenValues = {}

            openAPE_normalised = []
            openAPE_1_normalised = []
            openAPE_2_normalised  = []
            openAPE_3_normalised  = []

            closedAPE_normalised  = []
            closedAPE_1_normalised  = []
            closedAPE_2_normalised  = []
            closedAPE_3_normalised  = []

            openDict_normalised  = {}
            openDict_1_normalised  = {}
            openDict_2_normalised  = {}
            openDict_3_normalised  = {}

            closedDict_normalised  = {}
            closedDict_1_normalised  = {}
            closedDict_2_normalised  = {}
            closedDict_3_normalised  = {}

            for property, value in sentence['closedValues'].items():
                closedAPE_1.append(np.abs(value-extracted))
                closedAPE_2.append(np.abs(value-extracted)/(np.abs(extracted)+np.abs(value)))
                closedAPE_3.append(np.abs(value-extracted)/np.abs(extracted+value))

                closedDict_1[property] = (np.abs(value-extracted))
                closedDict_2[property] = (np.abs(value-extracted)/(np.abs(extracted)+np.abs(value)))
                closedDict_3[property] = (np.abs(value-extracted)/np.abs(extracted+value))

                normalisedClosedValues[property]=((value-property2rangeMin[property])/property2range[property])

            for property, value in sentence['openValues'].items():
                openAPE_1.append(np.abs(value-extracted))
                openAPE_2.append(np.abs(value-extracted)/(np.abs(extracted)+np.abs(value)))
                openAPE_3.append(np.abs(value-extracted)/np.abs(extracted+value))

                openDict_1[property] = (np.abs(value-extracted))
                openDict_2[property] = (np.abs(value-extracted)/(np.abs(extracted)+np.abs(value)))
                openDict_3[property] = (np.abs(value-extracted)/np.abs(extracted+value))

                normalisedOpenValues[property]=((value-property2rangeMin[property])/property2range[property])

            # print "normalisedOpenValues",normalisedOpenValues
            # print "normalisedClosedValues",normalisedClosedValues
            for property, value in normalisedClosedValues.items():
                closedAPE_normalised.append(np.abs(value-extractedNormal)/np.abs(extractedNormal))
                closedAPE_1_normalised.append(np.abs(value-extractedNormal))
                closedAPE_2_normalised.append(np.abs(value-extractedNormal)/(np.abs(extractedNormal)+np.abs(value)))
                closedAPE_3_normalised.append(np.abs(value-extractedNormal)/np.abs(extractedNormal+value))

                closedDict_normalised[property] = (np.abs(value-extractedNormal)/np.abs(extractedNormal))
                closedDict_1_normalised[property] = (np.abs(value-extractedNormal))
                closedDict_2_normalised[property] = (np.abs(value-extractedNormal)/(np.abs(extractedNormal)+np.abs(value)))
                closedDict_3_normalised[property] = (np.abs(value-extractedNormal)/np.abs(extractedNormal+value))

            for property, value in normalisedOpenValues.items():
                openAPE_normalised.append(np.abs(value-extractedNormal)/np.abs(extractedNormal))
                openAPE_1_normalised.append(np.abs(value-extractedNormal))
                openAPE_2_normalised.append(np.abs(value-extractedNormal)/(np.abs(extractedNormal)+np.abs(value)))
                openAPE_3_normalised.append(np.abs(value-extractedNormal)/np.abs(extractedNormal+value))

                openDict_normalised[property] = (np.abs(value-extractedNormal)/np.abs(extractedNormal))
                openDict_1_normalised[property] = (np.abs(value-extractedNormal))
                openDict_2_normalised[property] = (np.abs(value-extractedNormal)/(np.abs(extractedNormal)+np.abs(value)))
                openDict_3_normalised[property] = (np.abs(value-extractedNormal)/np.abs(extractedNormal+value))

            # Normalise the input value you see, and the whole value array to create costs

            #
            # openDict_1 = sorted(openDict_1.items(), key=operator.itemgetter(1))
            # openDict_2 = sorted(openDict_2.items(), key=operator.itemgetter(1))
            # openDict_3 = sorted(openDict_3.items(), key=operator.itemgetter(1))
            #
            # closedDict_1 = sorted(closedDict_1.items(), key=operator.itemgetter(1))
            # closedDict_2 = sorted(closedDict_2.items(), key=operator.itemgetter(1))
            # closedDict_3 = sorted(closedDict_3.items(), key=operator.itemgetter(1))

            # print "Value is ",sentence['location-value-pair']
            # print "Closed cost array is",sentence['closedCostArr']
            # print "Open cost array is",sentence['openCostArr']

            '''
            Get sorted versions of each array
            '''

            openAPE_1 = np.sort([value for i,value in enumerate(openAPE_1)])
            openAPE_2 = np.sort([value for i,value in enumerate(openAPE_2)])
            openAPE_3 = np.sort([value for i,value in enumerate(openAPE_3)])

            closedAPE_1 = np.sort([value for i,value in enumerate(closedAPE_1)])
            closedAPE_2 = np.sort([value for i,value in enumerate(closedAPE_2)])
            closedAPE_3 = np.sort([value for i,value in enumerate(closedAPE_3)])

            openAPE_normalised = np.sort([value for i,value in enumerate(openAPE_normalised)])
            openAPE_1_normalised = np.sort([value for i,value in enumerate(openAPE_1_normalised)])
            openAPE_2_normalised = np.sort([value for i,value in enumerate(openAPE_2_normalised)])
            openAPE_3_normalised = np.sort([value for i,value in enumerate(openAPE_3_normalised)])

            closedAPE_normalised = np.sort([value for i,value in enumerate(closedAPE_normalised)])
            closedAPE_1_normalised = np.sort([value for i,value in enumerate(closedAPE_1_normalised)])
            closedAPE_2_normalised = np.sort([value for i,value in enumerate(closedAPE_2_normalised)])
            closedAPE_3_normalised = np.sort([value for i,value in enumerate(closedAPE_3_normalised)])

            '''
            Get the chosen predictions and their minimum values - this is for the single array case.
            '''

            openPrediction = sentence['predictedPropertyOpen']
            closedPrediction = sentence['predictedPropertyClosed']

            openPrediction_normalised = min(openDict_normalised, key=openDict_normalised.get)
            closedPrediction_normalised = min(closedDict_normalised, key=closedDict_normalised.get)
            openAbsError_normalised = min(openAPE_normalised)
            closedAbsError_normalised = min(closedAPE_normalised)

            openPrediction_1 = min(openDict_1, key=openDict_1.get)
            closedPrediction_1 = min(closedDict_1, key=closedDict_1.get)
            openAbsError1 = min(openDict_1.itervalues())
            closedAbsError1 = min(closedDict_1.itervalues())

            openPrediction_1_normalised = min(openDict_1_normalised, key=openDict_1_normalised.get)
            closedPrediction_1_normalised = min(closedDict_1_normalised, key=closedDict_1_normalised.get)
            openAbsError1_normalised = min(openAPE_1_normalised)
            closedAbsError1_normalised = min(closedAPE_1_normalised)

            openPrediction_2 = min(openDict_2, key=openDict_2.get)
            closedPrediction_2 = min(closedDict_2, key=closedDict_2.get)
            openAbsError2 = min(openDict_2.itervalues())
            closedAbsError2 = min(closedDict_2.itervalues())

            openPrediction_2_normalised = min(openDict_2_normalised, key=openDict_2_normalised.get)
            closedPrediction_2_normalised = min(closedDict_2_normalised, key=closedDict_2_normalised.get)
            openAbsError2_normalised = min(openAPE_2_normalised)
            closedAbsError2_normalised = min(closedAPE_2_normalised)

            openPrediction_3 = min(openDict_3, key=openDict_3.get)
            closedPrediction_3 = min(closedDict_3, key=closedDict_3.get)
            openAbsError3 = min(openDict_3.itervalues())
            closedAbsError3 = min(closedDict_3.itervalues())

            openPrediction_3_normalised = min(openDict_3_normalised, key=openDict_3_normalised.get)
            closedPrediction_3_normalised = min(closedDict_3_normalised, key=closedDict_3_normalised.get)
            openAbsError3_normalised = min(openAPE_3_normalised)
            closedAbsError3_normalised = min(closedAPE_3_normalised)

            '''
            Get the remaining values
            '''
            openCostVectorRem = np.sort([value for i,value in enumerate(openAPE) if value is not sentence['meanAbsError']])
            closedCostVectorRem = np.sort([value for i,value in enumerate(closedAPE) if value is not sentence['closedMeanAbsError']])

            openCostVectorRem_1 = np.sort([value for i,value in enumerate(openAPE_1) if value is not openAbsError1])
            closedCostVectorRem_1 = np.sort([value for i,value in enumerate(closedAPE_1) if value is not closedAbsError1])

            openCostVectorRem_2 = np.sort([value for i,value in enumerate(openAPE_2) if value is not openAbsError2])
            closedCostVectorRem_2 = np.sort([value for i,value in enumerate(closedAPE_2) if value is not closedAbsError2])

            openCostVectorRem_3 = np.sort([value for i,value in enumerate(openAPE_3) if value is not openAbsError3])
            closedCostVectorRem_3 = np.sort([value for i,value in enumerate(closedAPE_3) if value is not closedAbsError3])

            '''
            Normalised versions of the remainder vectors
            '''

            openCostVectorRem_normalised = np.sort([value for i,value in enumerate(openAPE_normalised) if value is not openAbsError_normalised])
            closedCostVectorRem_normalised = np.sort([value for i,value in enumerate(closedAPE_normalised) if value is not closedAbsError_normalised])

            openCostVectorRem_1_normalised = np.sort([value for i,value in enumerate(openAPE_1_normalised) if value is not openAbsError1_normalised])
            closedCostVectorRem_1_normalised = np.sort([value for i,value in enumerate(closedAPE_1_normalised) if value is not closedAbsError1_normalised])

            openCostVectorRem_2_normalised = np.sort([value for i,value in enumerate(openAPE_2_normalised) if value is not openAbsError2_normalised])
            closedCostVectorRem_2_normalised = np.sort([value for i,value in enumerate(closedAPE_2_normalised) if value is not closedAbsError2_normalised])

            openCostVectorRem_3_normalised = np.sort([value for i,value in enumerate(openAPE_3_normalised) if value is not openAbsError3_normalised])
            closedCostVectorRem_3_normalised = np.sort([value for i,value in enumerate(closedAPE_3_normalised) if value is not closedAbsError3_normalised])

            '''
            Cost Function APE
            '''

            openIQR = float(np.subtract(*np.percentile(openAPE, [75, 25])))
            closedIQR = float(np.subtract(*np.percentile(closedAPE, [75, 25])))

            openMedian = float(np.median(openCostVectorRem))
            closedMedian = float(np.median(closedCostVectorRem))

            openGap = float(openAPE[1]-sentence['meanAbsError'])
            # print "Open gap is",openGap
            closedGap = float(closedAPE[1]-sentence['closedMeanAbsError'])
            # print "Closed gap is",closedGap

            closedRange = float(np.ptp(closedAPE))
            # print "Closed range is",closedRange

            openRange = float(np.ptp(openAPE))
            # print "Open range is",openRange
            if openRange==float(0):
                openRange=1e-10
            if closedRange==float(0):
                closedRange =1e-10

            # print "Open range is",openRange

            closedPercentArray = [float(val)/closedRange for val in closedAPE]
            openPercentArray = [float(val)/openRange for val in openAPE]

            # TODO - this is a hyperparameter
            closedCompetingEntries = float(sum(float(i) < float(1) for i in closedAPE))

            # print "Closed competing entries are",closedCompetingEntries
            openCompetingEntries = float(sum(float(i) < float(1) for i in openAPE))
            # print "openCompetingEntries are",openCompetingEntries

            # TODO - this is a hyperparameter
            closedCompetingPctEntries = float(sum(float(i) < float(0.05) for i in closedPercentArray))

            # print "Closed competing entries are",closedCompetingEntries
            openCompetingPctEntries = float(sum(float(i) < float(0.05) for i in openPercentArray))
            # print "openCompetingPctEntries are",openCompetingEntries

            closedGapPercent = float(closedGap/closedRange)
            # print "closedGapPercent is ",closedGapPercent

            openGapPercent = float(openGap/openRange)
            # print "openGapPercent is ",openGapPercent

            openSkewness = math.exp(st.stats.skew(openAPE, bias=False))

            closedSkewness = math.exp(st.stats.skew(closedAPE, bias=False))

            openMAD = meanabsdev(openAPE)

            '''
            Normalised version
            '''

            openIQR_normalised = float(np.subtract(*np.percentile(openAPE_normalised, [75, 25])))
            closedIQR_normalised = float(np.subtract(*np.percentile(closedAPE_normalised, [75, 25])))

            openMedian_normalised = float(np.median(openCostVectorRem_normalised))
            closedMedian_normalised = float(np.median(closedCostVectorRem_normalised))

            openGap_normalised = float(openAPE_normalised[1]-openAbsError_normalised)
            # print "Open gap is",openGap
            closedGap_normalised = float(closedAPE_normalised[1]-closedAbsError_normalised)
            # print "Closed gap is",closedGap

            closedRange_normalised = float(np.ptp(closedAPE_normalised))
            # print "Closed range is",closedRange

            openRange_normalised = float(np.ptp(openAPE_normalised))
            # print "Open range is",openRange
            if openRange_normalised==float(0):
                openRange_normalised=1e-10
            if closedRange_normalised==float(0):
                closedRange_normalised =1e-10

            # print "Open range is",openRange

            closedPercentArray_normalised = [float(val)/closedRange_normalised for val in closedAPE_normalised]
            openPercentArray_normalised = [float(val)/openRange_normalised for val in openAPE_normalised]

            # TODO - this is a hyperparameter
            closedCompetingEntries_normalised = float(sum(float(i) < float(1) for i in closedAPE_normalised))

            # print "Closed competing entries are",closedCompetingEntries
            openCompetingEntries_normalised = float(sum(float(i) < float(1) for i in openAPE_normalised))
            # print "openCompetingEntries are",openCompetingEntries

            # TODO - this is a hyperparameter
            closedCompetingPctEntries_normalised = float(sum(float(i) < float(0.05) for i in closedPercentArray_normalised))

            # print "Closed competing entries are",closedCompetingEntries
            openCompetingPctEntries_normalised = float(sum(float(i) < float(0.05) for i in openPercentArray_normalised))
            # print "openCompetingPctEntries are",openCompetingEntries

            closedGapPercent_normalised = float(closedGap_normalised/closedRange_normalised)
            # print "closedGapPercent is ",closedGapPercent

            openGapPercent_normalised = float(openGap_normalised/openRange_normalised)
            # print "openGapPercent is ",openGapPercent

            openSkewness_normalised = math.exp(st.stats.skew(openAPE_normalised, bias=False))

            closedSkewness_normalised = math.exp(st.stats.skew(closedAPE_normalised, bias=False))

            '''
            Cost Function 1
            '''

            openIQR_1 = float(np.subtract(*np.percentile(openAPE_1, [75, 25])))
            closedIQR_1 = float(np.subtract(*np.percentile(closedAPE_1, [75, 25])))

            openMedian_1 = float(np.median(openCostVectorRem_1))
            closedMedian_1 = float(np.median(closedCostVectorRem_1))

            openGap_1 = float(openAPE_1[1]-openAbsError1)
            # print "Open gap is",openGap
            closedGap_1 = float(closedAPE_1[1]-closedAbsError1)
            # print "Closed gap is",closedGap

            closedRange_1 = float(np.ptp(closedAPE_1))
            # print "Closed range is",closedRange

            openRange_1 = float(np.ptp(openAPE_1))
            # print "Open range is",openRange
            if openRange_1==float(0):
                openRange_1=1e-10
            if closedRange_1==float(0):
                closedRange_1 =1e-10

            # print "Open range is",openRange

            closedPercentArray_1 = [float(val)/closedRange_1 for val in closedAPE_1]
            openPercentArray_1 = [float(val)/openRange_1 for val in openAPE_1]

            # TODO - this is a hyperparameter
            closedCompetingEntries_1 = float(sum(float(i) < float(1) for i in closedAPE_1))

            # print "Closed competing entries are",closedCompetingEntries
            openCompetingEntries_1 = float(sum(float(i) < float(1) for i in openAPE_1))
            # print "openCompetingEntries are",openCompetingEntries

            # TODO - this is a hyperparameter
            closedCompetingPctEntries_1 = float(sum(float(i) < float(0.05) for i in closedPercentArray_1))

            # print "Closed competing entries are",closedCompetingEntries
            openCompetingPctEntries_1 = float(sum(float(i) < float(0.05) for i in openPercentArray_1))
            # print "openCompetingPctEntries are",openCompetingEntries

            closedGapPercent_1 = float(closedGap_1/closedRange_1)
            # print "closedGapPercent is ",closedGapPercent

            openGapPercent_1 = float(openGap_1/openRange_1)
            # print "openGapPercent is ",openGapPercent

            openSkewness_1 = math.exp(st.stats.skew(openAPE_1, bias=False))

            closedSkewness_1 = math.exp(st.stats.skew(closedAPE_1, bias=False))

            '''
            Normalised version
            '''

            openIQR_1_normalised = float(np.subtract(*np.percentile(openAPE_1_normalised, [75, 25])))
            closedIQR_1_normalised = float(np.subtract(*np.percentile(closedAPE_1_normalised, [75, 25])))

            openMedian_1_normalised = float(np.median(openCostVectorRem_1_normalised))
            closedMedian_1_normalised = float(np.median(closedCostVectorRem_1_normalised))

            openGap_1_normalised = float(openAPE_1_normalised[1]-openAbsError1_normalised)
            # print "Open gap is",openGap
            closedGap_1_normalised = float(closedAPE_1_normalised[1]-closedAbsError1_normalised)
            # print "Closed gap is",closedGap

            closedRange_1_normalised = float(np.ptp(closedAPE_1_normalised))
            # print "Closed range is",closedRange

            openRange_1_normalised = float(np.ptp(openAPE_1_normalised))
            # print "Open range is",openRange
            if openRange_1_normalised==float(0):
                openRange_1_normalised=1e-10
            if closedRange_1_normalised==float(0):
                closedRange_1_normalised =1e-10

            # print "Open range is",openRange

            closedPercentArray_1_normalised = [float(val)/closedRange_1_normalised for val in closedAPE_1_normalised]
            openPercentArray_1_normalised = [float(val)/openRange_1_normalised for val in openAPE_1_normalised]

            # TODO - this is a hyperparameter
            closedCompetingEntries_1_normalised = float(sum(float(i) < float(1) for i in closedAPE_1_normalised))

            # print "Closed competing entries are",closedCompetingEntries
            openCompetingEntries_1_normalised = float(sum(float(i) < float(1) for i in openAPE_1_normalised))
            # print "openCompetingEntries are",openCompetingEntries

            # TODO - this is a hyperparameter
            closedCompetingPctEntries_1_normalised = float(sum(float(i) < float(0.05) for i in closedPercentArray_1_normalised))

            # print "Closed competing entries are",closedCompetingEntries
            openCompetingPctEntries_1_normalised = float(sum(float(i) < float(0.05) for i in openPercentArray_1_normalised))
            # print "openCompetingPctEntries are",openCompetingEntries

            closedGapPercent_1_normalised = float(closedGap_1_normalised/closedRange_1_normalised)
            # print "closedGapPercent is ",closedGapPercent

            openGapPercent_1_normalised = float(openGap_1_normalised/openRange_1_normalised)
            # print "openGapPercent is ",openGapPercent

            openSkewness_1_normalised = math.exp(st.stats.skew(openAPE_1_normalised, bias=False))

            closedSkewness_1_normalised = math.exp(st.stats.skew(closedAPE_1_normalised, bias=False))

            '''
            Cost function 2
            '''

            openIQR_2 = float(np.subtract(*np.percentile(openAPE_2, [75, 25])))
            closedIQR_2 = float(np.subtract(*np.percentile(closedAPE_2, [75, 25])))

            openMedian_2 = float(np.median(openCostVectorRem_2))
            closedMedian_2 = float(np.median(closedCostVectorRem_2))

            openGap_2 = float(openAPE_2[1]-openAbsError2)
            # print "Open gap is",openGap
            closedGap_2 = float(closedAPE_2[1]-closedAbsError2)
            # print "Closed gap is",closedGap

            closedRange_2 = float(np.ptp(closedAPE_2))
            # print "Closed range is",closedRange

            openRange_2 = float(np.ptp(openAPE_2))
            # print "Open range is",openRange
            if openRange_2==float(0):
                openRange_2=1e-10
            if closedRange_2==float(0):
                closedRange_2 =1e-10

            # print "Open range is",openRange

            closedPercentArray_2 = [float(val)/closedRange_2 for val in closedAPE_2]
            openPercentArray_2 = [float(val)/openRange_2 for val in openAPE_2]

            # TODO - this is a hyperparameter
            closedCompetingEntries_2 = float(sum(float(i) < float(1) for i in closedAPE_2))

            # print "Closed competing entries are",closedCompetingEntries
            openCompetingEntries_2 = float(sum(float(i) < float(1) for i in openAPE_2))
            # print "openCompetingEntries are",openCompetingEntries

            # TODO - this is a hyperparameter
            closedCompetingPctEntries_2 = float(sum(float(i) < float(0.05) for i in closedPercentArray_2))

            # print "Closed competing entries are",closedCompetingEntries
            openCompetingPctEntries_2 = float(sum(float(i) < float(0.05) for i in openPercentArray_2))
            # print "openCompetingPctEntries are",openCompetingEntries

            closedGapPercent_2 = float(closedGap_2/closedRange_2)
            # print "closedGapPercent is ",closedGapPercent

            openGapPercent_2 = float(openGap_2/openRange_2)
            # print "openGapPercent is ",openGapPercent

            openSkewness_2 = math.exp(st.stats.skew(openAPE_2, bias=False))

            closedSkewness_2 = math.exp(st.stats.skew(closedAPE_2, bias=False))

            '''
            Normalised version
            '''

            openIQR_2_normalised = float(np.subtract(*np.percentile(openAPE_2_normalised, [75, 25])))
            closedIQR_2_normalised = float(np.subtract(*np.percentile(closedAPE_2_normalised, [75, 25])))

            openMedian_2_normalised = float(np.median(openCostVectorRem_2_normalised))
            closedMedian_2_normalised = float(np.median(closedCostVectorRem_2_normalised))

            openGap_2_normalised = float(openAPE_2_normalised[1]-openAbsError2_normalised)
            # print "Open gap is",openGap
            closedGap_2_normalised = float(closedAPE_2_normalised[1]-closedAbsError2_normalised)
            # print "Closed gap is",closedGap

            closedRange_2_normalised = float(np.ptp(closedAPE_2_normalised))
            # print "Closed range is",closedRange

            openRange_2_normalised = float(np.ptp(openAPE_2_normalised))
            # print "Open range is",openRange
            if openRange_2_normalised==float(0):
                openRange_2_normalised=1e-10
            if closedRange_2_normalised==float(0):
                closedRange_2_normalised =1e-10

            # print "Open range is",openRange

            closedPercentArray_2_normalised = [float(val)/closedRange_2_normalised for val in closedAPE_2_normalised]
            openPercentArray_2_normalised = [float(val)/openRange_2_normalised for val in openAPE_2_normalised]

            # TODO - this is a hyperparameter
            closedCompetingEntries_2_normalised = float(sum(float(i) < float(1) for i in closedAPE_2_normalised))

            # print "Closed competing entries are",closedCompetingEntries
            openCompetingEntries_2_normalised = float(sum(float(i) < float(1) for i in openAPE_2_normalised))
            # print "openCompetingEntries are",openCompetingEntries

            # TODO - this is a hyperparameter
            closedCompetingPctEntries_2_normalised = float(sum(float(i) < float(0.05) for i in closedPercentArray_2_normalised))

            # print "Closed competing entries are",closedCompetingEntries
            openCompetingPctEntries_2_normalised = float(sum(float(i) < float(0.05) for i in openPercentArray_2_normalised))
            # print "openCompetingPctEntries are",openCompetingEntries

            closedGapPercent_2_normalised = float(closedGap_2_normalised/closedRange_2_normalised)
            # print "closedGapPercent is ",closedGapPercent

            openGapPercent_2_normalised = float(openGap_2_normalised/openRange_2_normalised)
            # print "openGapPercent is ",openGapPercent

            openSkewness_2_normalised = math.exp(st.stats.skew(openAPE_2_normalised, bias=False))

            closedSkewness_2_normalised = math.exp(st.stats.skew(closedAPE_2_normalised, bias=False))


            '''
            Cost function 3
            '''

            openIQR_3 = float(np.subtract(*np.percentile(openAPE_3, [75, 25])))
            closedIQR_3 = float(np.subtract(*np.percentile(closedAPE_3, [75, 25])))

            openMedian_3 = float(np.median(openCostVectorRem_3))
            closedMedian_3 = float(np.median(closedCostVectorRem_3))

            openGap_3 = float(openAPE_3[1]-openAbsError3)
            # print "Open gap is",openGap
            closedGap_3 = float(closedAPE_3[1]-closedAbsError3)
            # print "Closed gap is",closedGap

            closedRange_3 = float(np.ptp(closedAPE_3))
            # print "Closed range is",closedRange

            openRange_3 = float(np.ptp(openAPE_3))
            # print "Open range is",openRange
            if openRange_3==float(0):
                openRange_3=1e-10
            if closedRange_3==float(0):
                closedRange_3 =1e-10

            # print "Open range is",openRange

            closedPercentArray_3 = [float(val)/closedRange_3 for val in closedAPE_3]
            openPercentArray_3 = [float(val)/openRange_3 for val in openAPE_3]

            # TODO - this is a hyperparameter
            closedCompetingEntries_3 = float(sum(float(i) < float(1) for i in closedAPE_3))

            # print "Closed competing entries are",closedCompetingEntries
            openCompetingEntries_3 = float(sum(float(i) < float(1) for i in openAPE_3))
            # print "openCompetingEntries are",openCompetingEntries

            # TODO - this is a hyperparameter
            closedCompetingPctEntries_3 = float(sum(float(i) < float(0.05) for i in closedPercentArray_3))

            # print "Closed competing entries are",closedCompetingEntries
            openCompetingPctEntries_3 = float(sum(float(i) < float(0.05) for i in openPercentArray_3))
            # print "openCompetingPctEntries are",openCompetingEntries

            closedGapPercent_3 = float(closedGap_3/closedRange_3)
            # print "closedGapPercent is ",closedGapPercent

            openGapPercent_3 = float(openGap_3/openRange_3)
            # print "openGapPercent is ",openGapPercent

            openSkewness_3 = math.exp(st.stats.skew(openAPE_3, bias=False))

            closedSkewness_3 = math.exp(st.stats.skew(closedAPE_3, bias=False))


            '''
            Normalised version
            '''

            openIQR_3_normalised = float(np.subtract(*np.percentile(openAPE_3_normalised, [75, 25])))
            closedIQR_3_normalised = float(np.subtract(*np.percentile(closedAPE_3_normalised, [75, 25])))

            openMedian_3_normalised = float(np.median(openCostVectorRem_3_normalised))
            closedMedian_3_normalised = float(np.median(closedCostVectorRem_3_normalised))

            openGap_3_normalised = float(openAPE_3_normalised[1]-openAbsError3_normalised)
            # print "Open gap is",openGap
            closedGap_3_normalised = float(closedAPE_3_normalised[1]-closedAbsError3_normalised)
            # print "Closed gap is",closedGap

            closedRange_3_normalised = float(np.ptp(closedAPE_3_normalised))
            # print "Closed range is",closedRange

            openRange_3_normalised = float(np.ptp(openAPE_3_normalised))
            # print "Open range is",openRange
            if openRange_3_normalised==float(0):
                openRange_3_normalised=1e-10
            if closedRange_3_normalised==float(0):
                closedRange_3_normalised =1e-10

            # print "Open range is",openRange

            closedPercentArray_3_normalised = [float(val)/closedRange_3_normalised for val in closedAPE_3_normalised]
            openPercentArray_3_normalised = [float(val)/openRange_3_normalised for val in openAPE_3_normalised]

            # TODO - this is a hyperparameter
            closedCompetingEntries_3_normalised = float(sum(float(i) < float(1) for i in closedAPE_3_normalised))

            # print "Closed competing entries are",closedCompetingEntries
            openCompetingEntries_3_normalised = float(sum(float(i) < float(1) for i in openAPE_3_normalised))
            # print "openCompetingEntries are",openCompetingEntries

            # TODO - this is a hyperparameter
            closedCompetingPctEntries_3_normalised = float(sum(float(i) < float(0.05) for i in closedPercentArray_3_normalised))

            # print "Closed competing entries are",closedCompetingEntries
            openCompetingPctEntries_3_normalised = float(sum(float(i) < float(0.05) for i in openPercentArray_3_normalised))
            # print "openCompetingPctEntries are",openCompetingEntries

            closedGapPercent_3_normalised = float(closedGap_3_normalised/closedRange_3_normalised)
            # print "closedGapPercent is ",closedGapPercent

            openGapPercent_3_normalised = float(openGap_3_normalised/openRange_3_normalised)
            # print "openGapPercent is ",openGapPercent

            openSkewness_3_normalised = math.exp(st.stats.skew(openAPE_3_normalised, bias=False))

            closedSkewness_3_normalised = math.exp(st.stats.skew(closedAPE_3_normalised, bias=False))


            '''
            Use calculations on the normalised format for the single version, and normalise also
            '''

            for key, data in costMatrix.iteritems():
                # When there is only one cost per training instance
                if str(key).startswith("single"):
                    if str(key).split('_')[1]=="open":
                        if str(key).endswith("1"):
                            data['cost_matrix'].append(error)
                            # Get normalised version and different cost matrices versions of each one, then calculations based on the normalised format
                        if str(key).endswith("2"):
                            data['cost_matrix'].append(error/float(openIQR*(len(openCostVectorRem)+1)))
                        if str(key).endswith("3"):
                            data['cost_matrix'].append(error/float(openMedian*(len(openCostVectorRem)+1)))
                        if str(key).endswith("4"):
                            data['cost_matrix'].append(error/float(openIQR))

                        if str(key).endswith("5"):
                            data['cost_matrix'].append(error/float(openMedian))

                        if str(key).endswith("6"):
                            data['cost_matrix'].append(error/np.where(openGap
    >0,openGap, 0e-10))
                        if str(key).endswith("7"):
                            data['cost_matrix'].append(error/np.where(openGapPercent
    >0,openGapPercent, 0e-10))
                        if str(key).endswith("8"):
                            data['cost_matrix'].append(error*openCompetingEntries)
                        if str(key).endswith("9"):
                            data['cost_matrix'].append(error*openCompetingPctEntries)
                        if str(key).endswith("10"):
                            data['cost_matrix'].append(0)
                        # Do calculations on the normalised versions
                        if str(key).endswith("11"):
                            data['cost_matrix'].append(openAbsError_normalised)
                        if str(key).endswith("12"):
                            data['cost_matrix'].append(openAbsError_normalised/float(openIQR_normalised*(len(openCostVectorRem_normalised)+1)))
                        if str(key).endswith("13"):
                            data['cost_matrix'].append(openAbsError_normalised/float(openMedian_normalised*(len(openCostVectorRem_normalised)+1)))
                        if str(key).endswith("14"):
                            data['cost_matrix'].append(openAbsError_normalised/float(openIQR_normalised))

                        if str(key).endswith("15"):
                            data['cost_matrix'].append(openAbsError_normalised/float(openMedian_normalised))

                        if str(key).endswith("16"):
                            data['cost_matrix'].append(openAbsError_normalised/np.where(openGap_normalised
    >0,openGap_normalised, 0e-10))
                        if str(key).endswith("17"):
                            data['cost_matrix'].append(openAbsError_normalised/np.where(openGapPercent
    >0,openGapPercent_normalised, 0e-10))
                        if str(key).endswith("18"):
                            data['cost_matrix'].append(openAbsError_normalised*openCompetingEntries_normalised)
                        if str(key).endswith("19"):
                            data['cost_matrix'].append(openAbsError_normalised*openCompetingPctEntries_normalised)
                        if str(key).endswith("20"):
                            data['cost_matrix'].append(openAbsError1)
                            # Get normalised version and different cost matrices versions of each one, then calculations based on the normalised format
                        if str(key).endswith("21"):
                            data['cost_matrix'].append(openAbsError1/float(openIQR*(len(openCostVectorRem)+1)))
                        if str(key).endswith("22"):
                            data['cost_matrix'].append(openAbsError1/float(openMedian*(len(openCostVectorRem)+1)))
                        if str(key).endswith("23"):
                            data['cost_matrix'].append(openAbsError1/float(openIQR))

                        if str(key).endswith("24"):
                            data['cost_matrix'].append(openAbsError1/float(openMedian))

                        if str(key).endswith("25"):
                            data['cost_matrix'].append(openAbsError1/np.where(openGap
    >0,openGap, 0e-10))
                        if str(key).endswith("26"):
                            data['cost_matrix'].append(openAbsError1/np.where(openGapPercent
    >0,openGapPercent, 0e-10))
                        if str(key).endswith("27"):
                            data['cost_matrix'].append(openAbsError1*openCompetingEntries)
                        if str(key).endswith("28"):
                            data['cost_matrix'].append(openAbsError1*openCompetingPctEntries)
                        if str(key).endswith("29"):
                            data['cost_matrix'].append(openAbsError1_normalised)
                        if str(key).endswith("30"):
                            data['cost_matrix'].append(openAbsError1_normalised/float(openIQR_normalised*(len(openCostVectorRem_normalised)+1)))
                        if str(key).endswith("31"):
                            data['cost_matrix'].append(openAbsError1_normalised/float(openMedian_normalised*(len(openCostVectorRem_normalised)+1)))
                        if str(key).endswith("32"):
                            data['cost_matrix'].append(openAbsError1_normalised/float(openIQR_normalised))

                        if str(key).endswith("33"):
                            data['cost_matrix'].append(openAbsError1_normalised/float(openMedian_normalised))

                        if str(key).endswith("34"):
                            data['cost_matrix'].append(openAbsError1_normalised/np.where(openGap_normalised
    >0,openGap_normalised, 0e-10))
                        if str(key).endswith("35"):
                            data['cost_matrix'].append(openAbsError1_normalised/np.where(openGapPercent
    >0,openGapPercent_normalised, 0e-10))
                        if str(key).endswith("36"):
                            data['cost_matrix'].append(openAbsError1_normalised*openCompetingEntries_normalised)
                        if str(key).endswith("37"):
                            data['cost_matrix'].append(openAbsError1_normalised*openCompetingPctEntries_normalised)
                        if str(key).endswith("38"):
                            data['cost_matrix'].append(openAbsError2)
                        if str(key).endswith("39"):
                            data['cost_matrix'].append(openAbsError2/float(openIQR*(len(openCostVectorRem)+1)))
                        if str(key).endswith("40"):
                            data['cost_matrix'].append(openAbsError2/float(openMedian*(len(openCostVectorRem)+1)))
                        if str(key).endswith("41"):
                            data['cost_matrix'].append(openAbsError2/float(openIQR))
                        if str(key).endswith("42"):
                            data['cost_matrix'].append(openAbsError2/float(openMedian))
                        if str(key).endswith("43"):
                            data['cost_matrix'].append(openAbsError2/np.where(openGap
    >0,openGap, 0e-10))
                        if str(key).endswith("44"):
                            data['cost_matrix'].append(openAbsError2/np.where(openGapPercent
    >0,openGapPercent, 0e-10))
                        if str(key).endswith("45"):
                            data['cost_matrix'].append(openAbsError2*openCompetingEntries)
                        if str(key).endswith("46"):
                            data['cost_matrix'].append(openAbsError2*openCompetingPctEntries)
                        if str(key).endswith("47"):
                            data['cost_matrix'].append(openAbsError2_normalised)
                        if str(key).endswith("48"):
                            data['cost_matrix'].append(openAbsError2_normalised/float(openIQR_normalised*(len(openCostVectorRem_normalised)+1)))
                        if str(key).endswith("49"):
                            data['cost_matrix'].append(openAbsError2_normalised/float(openMedian_normalised*(len(openCostVectorRem_normalised)+1)))
                        if str(key).endswith("50"):
                            data['cost_matrix'].append(openAbsError2_normalised/float(openIQR_normalised))

                        if str(key).endswith("51"):
                            data['cost_matrix'].append(openAbsError2_normalised/float(openMedian_normalised))

                        if str(key).endswith("52"):
                            data['cost_matrix'].append(openAbsError2_normalised/np.where(openGap_normalised
    >0,openGap_normalised, 0e-10))
                        if str(key).endswith("53"):
                            data['cost_matrix'].append(openAbsError2_normalised/np.where(openGapPercent
    >0,openGapPercent_normalised, 0e-10))
                        if str(key).endswith("54"):
                            data['cost_matrix'].append(openAbsError2_normalised*openCompetingEntries_normalised)
                        if str(key).endswith("55"):
                            data['cost_matrix'].append(openAbsError2_normalised*openCompetingPctEntries_normalised)
                        if str(key).endswith("56"):
                            data['cost_matrix'].append(openAbsError3)
                        if str(key).endswith("57"):
                            data['cost_matrix'].append(openAbsError3/float(openIQR*(len(openCostVectorRem)+1)))
                        if str(key).endswith("58"):
                            data['cost_matrix'].append(openAbsError3/float(openMedian*(len(openCostVectorRem)+1)))
                        if str(key).endswith("59"):
                            data['cost_matrix'].append(openAbsError3/float(openIQR))
                        if str(key).endswith("60"):
                            data['cost_matrix'].append(openAbsError3/float(openMedian))
                        if str(key).endswith("61"):
                            data['cost_matrix'].append(openAbsError3/np.where(openGap
    >0,openGap, 0e-10))
                        if str(key).endswith("62"):
                            data['cost_matrix'].append(openAbsError3/np.where(openGapPercent
    >0,openGapPercent, 0e-10))
                        if str(key).endswith("63"):
                            data['cost_matrix'].append(openAbsError3*openCompetingEntries)
                        if str(key).endswith("64"):
                            data['cost_matrix'].append(openAbsError3*openCompetingPctEntries)
                        if str(key).endswith("65"):
                            data['cost_matrix'].append(openAbsError3_normalised)
                        if str(key).endswith("66"):
                            data['cost_matrix'].append(openAbsError3_normalised/float(openIQR_normalised*(len(openCostVectorRem_normalised)+1)))
                        if str(key).endswith("67"):
                            data['cost_matrix'].append(openAbsError3_normalised/float(openMedian_normalised*(len(openCostVectorRem_normalised)+1)))
                        if str(key).endswith("68"):
                            data['cost_matrix'].append(openAbsError3_normalised/float(openIQR_normalised))

                        if str(key).endswith("69"):
                            data['cost_matrix'].append(openAbsError3_normalised/float(openMedian_normalised))

                        if str(key).endswith("70"):
                            data['cost_matrix'].append(openAbsError3_normalised/np.where(openGap_normalised
    >0,openGap_normalised, 0e-10))
                        if str(key).endswith("71"):
                            data['cost_matrix'].append(openAbsError3_normalised/np.where(openGapPercent
    >0,openGapPercent_normalised, 0e-10))
                        if str(key).endswith("72"):
                            data['cost_matrix'].append(openAbsError3_normalised*openCompetingEntries_normalised)
                        if str(key).endswith("73"):
                            data['cost_matrix'].append(openAbsError3_normalised*openCompetingPctEntries_normalised)
                        # Now we normalise everything row-wise (not the normalised versions)
                        if str(key).endswith("74"):
                            data['cost_matrix'].append(error/openRange)
                        if str(key).endswith("75"):
                            data['cost_matrix'].append(openAbsError1/openRange_1)
                        if str(key).endswith("76"):
                            data['cost_matrix'].append(openAbsError2/openRange_2)
                        if str(key).endswith("77"):
                            data['cost_matrix'].append(openAbsError3/openRange_3)
                        if str(key).endswith("78"):
                            data['cost_matrix'].append(openAbsError_normalised/openRange_normalised)
                        if str(key).endswith("79"):
                            data['cost_matrix'].append(openAbsError1_normalised/openRange_1_normalised)
                        if str(key).endswith("80"):
                            data['cost_matrix'].append(openAbsError2_normalised/openRange_2_normalised)
                        if str(key).endswith("81"):
                            data['cost_matrix'].append(openAbsError3_normalised/openRange_3_normalised)
                        # Now multiply everything by the skewness score, as positive skewness
                        if str(key).endswith("82"):
                            data['cost_matrix'].append(error*openSkewness)
                        if str(key).endswith("83"):
                            data['cost_matrix'].append(openAbsError1*openSkewness_1)
                        if str(key).endswith("84"):
                            data['cost_matrix'].append(openAbsError2*openSkewness_2)
                        if str(key).endswith("85"):
                            data['cost_matrix'].append(openAbsError3*openSkewness_3)
                        if str(key).endswith("86"):
                            data['cost_matrix'].append(openAbsError_normalised*openSkewness_normalised)
                        if str(key).endswith("87"):
                            data['cost_matrix'].append(openAbsError1_normalised*openSkewness_1_normalised)
                        if str(key).endswith("88"):
                            data['cost_matrix'].append(openAbsError2_normalised*openSkewness_2_normalised)
                        if str(key).endswith("89"):
                            data['cost_matrix'].append(openAbsError3_normalised*openSkewness_3_normalised)
                    else:
                        if str(key).endswith("1"):
                            data['cost_matrix'].append(closedError)
                        if str(key).endswith("2"):
                            data['cost_matrix'].append(closedError/float(closedIQR*(len(closedCostVectorRem)+1)))
                        if str(key).endswith("3"):
                            data['cost_matrix'].append(closedError/float(closedMedian*(len(closedCostVectorRem)+1)))
                        if str(key).endswith("4"):
                            data['cost_matrix'].append(closedError/np.where(closedGap>0,closedGap,1e-10))
                        if str(key).endswith("5"):
                            data['cost_matrix'].append(closedError/np.where(closedGapPercent>0,closedGapPercent,1e-10))
                        if str(key).endswith("6"):
                            data['cost_matrix'].append(closedError*closedCompetingEntries)
                        if str(key).endswith("7"):
                            data['cost_matrix'].append(closedError*closedCompetingPctEntries)
                        if str(key).endswith("8"):
                            data['cost_matrix'].append(closedError*closedCompetingPctEntries)
                else:
                    if str(key).split('_')[0]=="open":
                        if str(key).endswith("1"):
                            # dict = {}
                            # for key,value in openDict.iteritems():
                            #     dict[key] = value/openRange
                            # array.append(dict)
                            # array.append(openDict)
                            # dict = {}
                            # for key,value in openDict.iteritems():
                            #     if key == openPrediction:
                            #         dict[key] = 0
                            #     else:
                            #         dict[key] = 1
                            # array.append(dict)
                            dict = {}
                            for i,key in enumerate(openKeySet):
                                if key == openPrediction:
                                    dict[key] = 0
                                else:
                                    dict[key] = 1
                            data['cost_matrix'].append(dict)
                        if str(key).endswith("2"):
                            dict = {}
                            for key,value in openDict.iteritems():
                                dict[key] = value/float(openIQR*(len(openCostVectorRem)+1))
                            data['cost_matrix'].append(dict)
                        if str(key).endswith("3"):
                            dict = {}
                            for key,value in openDict.iteritems():
                                dict[key] = value/float(openMedian*(len(openCostVectorRem)+1))
                            data['cost_matrix'].append(dict)
                        if str(key).endswith("4"):
                            dict = {}
                            for key,value in openDict.iteritems():
                                dict[key] = value/np.where(openGap
    >0,openGap, 0e-10)
                            data['cost_matrix'].append(dict)
                        if str(key).endswith("5"):
                            dict = {}
                            for key,value in openDict.iteritems():
                                dict[key] = value/np.where(openGapPercent
    >0,openGapPercent, 0e-10)
                            data['cost_matrix'].append(dict)
                        if str(key).endswith("6"):
                            dict = {}
                            for key,value in openDict.iteritems():
                                dict[key] = value*openCompetingEntries
                            data['cost_matrix'].append(dict)
                        if str(key).endswith("7"):
                            dict = {}
                            for key,value in openDict.iteritems():
                                dict[key] = value*openCompetingPctEntries
                            data['cost_matrix'].append(dict)
                        if str(key).endswith("7"):
                            dict = {}
                            for key,value in openDict.iteritems():
                                dict[key] = value*openCompetingPctEntries
                            data['cost_matrix'].append(dict)
                        if str(key).endswith("8"):
                            dict = {}
                            for key,value in openDict.iteritems():
                                if key == openPrediction:
                                    dict[key] = 0
                                else:
                                    dict[key] = 1
                            data['cost_matrix'].append(dict)
                    else:
                        if str(key).endswith("1"):
                            # dict = {}
                            # for key,value in closedDict.iteritems():
                            #     if key == closedPrediction:
                            #         dict[key] = 0
                            #     else:
                            #         dict[key] = 1
                            # array.append(dict)
                            dict = {}
                            for i,key in enumerate(closedKeySet):
                                if key == closedPrediction:
                                    dict[key] = 0
                                else:
                                    dict[key] = 1
                            data['cost_matrix'].append(dict)
                            # dict = {}
                            # for key,value in closedDict.iteritems():
                            #     dict[key] = value/closedRange
                            # array.append(dict)
                            # array.append(closedDict)
                        if str(key).endswith("2"):
                            dict = {}
                            for key,value in closedDict.iteritems():
                                dict[key] = value/float(closedIQR*(len(closedCostVectorRem)+1))
                            data['cost_matrix'].append(dict)
                        if str(key).endswith("3"):
                            dict = {}
                            for key,value in closedDict.iteritems():
                                dict[key] = value/float(closedMedian*(len(closedCostVectorRem)+1))
                            data['cost_matrix'].append(dict)
                        if str(key).endswith("4"):
                            dict = {}
                            for key,value in closedDict.iteritems():
                                dict[key] = value/np.where(closedGap>0,closedGap,0e-10)
                            data['cost_matrix'].append(dict)
                        if str(key).endswith("5"):
                            dict = {}
                            for key,value in closedDict.iteritems():
                                dict[key] = value/np.where(closedGapPercent>0,closedGapPercent,0e-10)
                            data['cost_matrix'].append(dict)
                        if str(key).endswith("6"):
                            dict = {}
                            for key,value in closedDict.iteritems():
                                dict[key] = value*closedCompetingEntries
                            data['cost_matrix'].append(dict)
                        if str(key).endswith("7"):
                            dict = {}
                            for key,value in closedDict.iteritems():
                                dict[key] = value*closedCompetingPctEntries
                            data['cost_matrix'].append(dict)
                        if str(key).endswith("8"):
                            dict = {}
                            for key,value in closedDict.iteritems():
                                if key == closedPrediction:
                                    dict[key] = 0
                                else:
                                    dict[key] = 1
                            data['cost_matrix'].append(dict)

    '''
    Normalise everything in different ways
    '''




# values = sum([[item["value"]] + item["openValues"].values()
#               for item in array], [])
# v_min, v_max = min(values), max(values)
# output = [f(v_min, v_max, item) for item in array]
# print output

    # for key, value in costMatrix.iteritems():
    #     print key," array is ", value['cost_matrix'],"\n"

def generateDatFiles(costDict, trainingLabels, closedTrainingLabels, trainingFeatures,bigramFeatures, depgramFeatures,wordgramFeatures):

    # TODO - make sure this opens a fresh file all the time and replaces

    print "There are this many unique labels",len(set(trainingLabels))
    print "There are this many closed unique labels",len(set(closedTrainingLabels))

    print "Generating VW files...\n"

    for model,dict in costDict.items():

        vowpal_file = open(dict['words_path'], 'w')
        arow_file = open(dict['words_arow_path'], 'w')
        ld_file = open(dict['words_ld_path'], 'w')

        vowpal_file_bigrams = open(dict['bigrams_path'], 'w')
        arow_file_bigrams = open(dict['bigrams_arow_path'], 'w')
        ld_file_bigrams = open(dict['bigrams_ld_path'], 'w')

        vowpal_file_depgrams = open(dict['depgrams_path'], 'w')
        arow_file_depgrams = open(dict['depgrams_arow_path'], 'w')
        ld_file_depgrams = open(dict['depgrams_ld_path'], 'w')

        vowpal_file_wordgrams = open(dict['wordgrams_path'], 'w')
        arow_file_wordgrams = open(dict['wordgrams_arow_path'], 'w')
        ld_file_wordgrams = open(dict['wordgrams_ld_path'], 'w')

        for i, (label,closedLabel,costDict, features,bigrams, depgrams,worgrams) in enumerate(zip(trainingLabels,closedTrainingLabels,dict['cost_matrix'],trainingFeatures,bigramFeatures,depgramFeatures,wordgramFeatures)):

            depgrams = depgrams.decode('utf-8')

            # print "Types are",type(features),type(bigrams),type(depgrams),type(worgrams)
            arow_line = ""
            vowpal_line = ""
            ld_line = ""

            arow_line_bigrams = ""
            vowpal_line_bigrams = ""
            ld_line_bigrams = ""

            arow_line_depgrams = ""
            vowpal_line_depgrams = ""
            ld_line_depgrams = ""

            arow_line_wordgrams = ""
            vowpal_line_wordgrams = ""
            ld_line_wordgrams = ""

            # print label, closedLabel,costDict,features
            if model.startswith("single"):
                # If its a single model within only cost, costDict is just one value
                if str(model).split('_')[1]=="open":
                    # features for vowpal format only
                    # + " " + str(i) +
                    #  (" ").join(map(str, features))
                    vowpal_line+= str(label) + ":" + str(costDict) + " " + str(i) + " | " + features
                    arow_line += str(label) + ":" + str(costDict) + " | " + features
                    arow_file.write(arow_line+"\n")

                    print "Types are",type(bigrams),type(depgrams),type(wordgrams)

                    vowpal_line_bigrams+= str(label) + ":" + str(costDict) + " " + str(i) + " | " + bigrams
                    arow_line_bigrams += str(label) + ":" + str(costDict) + " | " + bigrams
                    arow_file_bigrams.write(arow_line_bigrams+"\n")

                    vowpal_line_depgrams+= str(label) + ":" + str(costDict) + " " + str(i) + " | " + depgrams.encode('utf-8')
                    arow_line_depgrams += str(label) + ":" + str(costDict) + " | " + depgrams.encode('utf-8')
                    arow_file_depgrams.write(arow_line_depgrams+"\n")

                    vowpal_line_wordgrams+= str(label) + ":" + str(costDict) + " " + str(i) + " | " + worgrams.encode('utf-8')
                    arow_line_wordgrams += str(label) + ":" + str(costDict) + " | " + worgrams.encode('utf-8')
                    arow_file_wordgrams.write(arow_line_wordgrams+"\n")


                else:
                    vowpal_line+= str(closedLabel) + ":" + str(costDict) + " " + str(i) + " | " + features
                    arow_line += str(closedLabel) + ":" + str(costDict) + " | " + features
                    arow_file.write(arow_line+"\n")

                    vowpal_line_bigrams+= str(closedLabel) + ":" + str(costDict) + " " + str(i) + " | " + bigrams
                    arow_line_bigrams += str(closedLabel) + ":" + str(costDict) + " | " + bigrams
                    arow_file_bigrams.write(arow_line_bigrams+"\n")

                    vowpal_line_depgrams+= str(closedLabel) + ":" + str(costDict) + " " + str(i) + " | " + depgrams.encode('utf-8')
                    arow_line_depgrams += str(closedLabel) + ":" + str(costDict) + " | " + depgrams.encode('utf-8')
                    arow_file_depgrams.write(arow_line_depgrams+"\n")

                    vowpal_line_wordgrams+= str(closedLabel) + ":" + str(costDict) + " " + str(i) + " | " + worgrams.encode('utf-8')
                    arow_line_wordgrams += str(closedLabel) + ":" + str(costDict) + " | " + worgrams.encode('utf-8')
                    arow_file_wordgrams.write(arow_line_wordgrams+"\n")
            else:
                # if str(model).split('_')[1]=="open":
                #
                # else:
                ld_line += "shared | " + features + "\n"
                ld_line_bigrams += "shared | " + bigrams + "\n"
                ld_line_depgrams += "shared | " + depgrams.encode('utf-8') + "\n"
                ld_line_wordgrams += "shared | " + worgrams.encode('utf-8') + "\n"

                for count,(lb, cost) in enumerate(costDict.items()):
                     # + " "
                    vowpal_line += str(lb) + ":" + str(cost) + " "
                    arow_line += str(lb) + ":" + str(cost) + " "
                    ld_line+=str(lb) + ":" + str(cost) + " | " + "label"+str(count+1)+"\n"
                    ld_line_bigrams+=str(lb) + ":" + str(cost) + " | " + "label"+str(count+1)+"\n"
                    ld_line_depgrams+=str(lb) + ":" + str(cost) + " | " + "label"+str(count+1)+"\n"
                    ld_line_wordgrams+=str(lb) + ":" + str(cost) + " | " + "label"+str(count+1)+"\n"
                # str(i) +
                # (" ").join(map(str, features))

                vowpal_line += str(i) + "| " + features
                arow_line += "| " + features

                vowpal_line_bigrams += str(i) + "| " + bigrams
                arow_line_bigrams += "| " + bigrams

                vowpal_line_depgrams += str(i) + "| " + depgrams.encode('utf-8')
                arow_line_depgrams += "| " + depgrams.encode('utf-8')

                vowpal_line_wordgrams += str(i) + "| " + worgrams.encode('utf-8')
                arow_line_wordgrams += "| " + worgrams.encode('utf-8')

                # Now write to files
                vowpal_file.write(vowpal_line +"\n")
                arow_file.write(arow_line+"\n")
                ld_file.write(ld_line+"\n")

                vowpal_file_bigrams.write(vowpal_line_bigrams +"\n")
                arow_file_bigrams.write(arow_line_bigrams +"\n")
                ld_file_bigrams.write(ld_line_bigrams +"\n")

                vowpal_file_depgrams.write(vowpal_line_depgrams +"\n")
                arow_file_depgrams.write(arow_line_depgrams +"\n")
                ld_file_depgrams.write(ld_line_depgrams +"\n")

                vowpal_file_wordgrams.write(vowpal_line_wordgrams +"\n")
                arow_file_wordgrams.write(arow_line_wordgrams +"\n")
                ld_file_wordgrams.write(ld_line_wordgrams +"\n")


def find_ngrams(input_list, n):
  return zip(*[input_list[i:] for i in range(n)])


def training_features(inputSentences):
    global vectorizer
    global train_wordbigram_list_sparse
    global train_bigram_list_sparse
    global train_wordlist_sparse
    global train_gramlist_sparse

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

            wordgrams = find_ngrams(word_list,2)
            for i,grams in enumerate(wordgrams):
              wordgrams[i] = '+'.join(grams)
            wordgrams= (" ").join(wordgrams)
            train_gramlist.append(wordgrams)

            train_property_labels.append(sentence['predictedPropertyOpen'])
            train_property_labels_threshold.append(sentence['predictedPropertyOpenThreshold'])
            closed_train_property_labels.append(sentence['predictedPropertyClosed'])
            closed_train_property_labels_threshold.append(sentence['predictedPropertyClosedThreshold'])

    # print "These are the clean words in the training sentences: ", train_wordlist
    # print "These are the labels in the training sentences: ", train_labels
    print "Creating the bag of words...\n"
    train_wordlist_sparse = vectorizer.fit_transform(train_wordlist).toarray().astype(np.float)
    train_gramlist_sparse = vectorizer.fit_transform(train_gramlist).toarray().astype(np.float)
    train_bigram_list_sparse = vectorizer.fit_transform(train_bigram_list).toarray().astype(np.float)
    train_wordbigram_list_sparse = vectorizer.fit_transform(train_wordbigram_list).toarray().astype(np.float)


def test_features(testSentences):
    global vectorizer
    global test_gramlist_sparse
    global test_bigram_list_sparse
    global test_wordbigram_list_sparse
    global test_wordlist_sparse

    for sentence in testSentences:
        if sentence['parsedSentence'] != {} and sentence['mape_label'] != {}:
            words = " ".join(sentence_to_words(sentence['parsedSentence'], True))
            word_list = sentence_to_words(sentence['parsedSentence'], True)
            wordgrams = find_ngrams(word_list,2)
            for i,grams in enumerate(wordgrams):
              wordgrams[i] = '+'.join(grams)
            wordgrams= (" ").join(wordgrams)
            test_gramlist.append(wordgrams)
            test_wordlist.append(words)
            test_property_labels.append(sentence['property'])
            bigrams = sentence['depPath']
            # print "Test Bigrams are",bigrams
            test_bigram_list.append(bigrams)
            test_wordbigram_list.append(words + " " + bigrams.decode("utf-8"))
    # print "These are the clean words in the test sentences: ", clean_test_sentences
    # print "These are the mape labels in the test sentences: ", binary_test_labels
    test_wordlist_sparse = vectorizer.fit_transform(test_wordlist).toarray().astype(np.float)
    test_wordbigram_list_sparse = vectorizer.fit_transform(test_wordbigram_list).toarray().astype(np.float)
    test_bigram_list_sparse = vectorizer.fit_transform(test_bigram_list).toarray().astype(np.float)
    test_gramlist_sparse = vectorizer.fit_transform(test_gramlist).toarray().astype(np.float)

    # test_data_features = vectorizer.transform(clean_test_sentences)
    # test_data_features = test_data_features.toarray()
    # test_data_features = test_data_features.astype(np.float)

if __name__ == "__main__":
    # training data
    # load the sentence file for training

    property2region2value = loadMatrix(sys.argv[1])

    with open(sys.argv[2]) as trainingSentences:
        pattern2regions = json.loads(trainingSentences.read())

    print "We have ", len(pattern2regions), " training sentences."
    # We load in the allowable features and also no_region

    with open(sys.argv[3]) as testSentences:
        testSentences = json.loads(testSentences.read())

    threshold = testSentences[0]['threshold']

    print "Threshold is ",threshold

    with open(sys.argv[4]) as featuresKept:
        properties = json.loads(featuresKept.read())
    # properties.append("no_region")

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

    train_wordlist_sparse = []
    test_wordlist_sparse = []

    train_gramlist_sparse = []
    test_gramlist_sparse = []

    train_bigram_list_sparse = []
    test_bigram_list_sparse = []

    train_wordbigram_list_sparse = []
    test_wordbigram_list_sparse = []

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

    vectorizer = Pipeline([('vect', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                          ('tfidf', TfidfTransformer(use_idf=True,norm='l2',sublinear_tf=True))])

    print "Getting all the words in the training sentences...\n"

    # This both sorts out the features and the training labels
    training_features(pattern2regions)

    print len(train_wordlist), "sets of training features"


    trainingLabels = len(pattern2regions)
    positiveOpenTrainingLabels = len(train_property_labels_threshold) - train_property_labels_threshold.count(
        "no_region")
    negativeOpenTrainingLabels = train_property_labels_threshold.count("no_region")
    positiveClosedTrainingLabels = len(closed_train_property_labels_threshold) - closed_train_property_labels_threshold.count(
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

    # TODO - this should be one matrix with different keys
    cost_matrices = {}

    for x in range(1,2):
        cost_matrices["open_cost_{0}".format(x)]={'cost_matrix':[]}
        cost_matrices["closed_cost_{0}".format(x)]={'cost_matrix':[]}
        cost_matrices["single_open_cost_{0}".format(x)]={'cost_matrix':[]}
        cost_matrices["single_closed_cost_{0}".format(x)]={'cost_matrix':[]}
        # cost_matrices["normalised_open_cost_{0}".format(x)]=[]
        # cost_matrices["normalised_closed_cost_{0}".format(x)]=[]


    model_path = os.path.join(sys.argv[5] +'/models.txt')

    with open(model_path, "wb") as out:
        json.dump(cost_matrices, out,indent=4)

    openKeySet = set(train_property_labels)
    openMapping = {val:key+1 for key,val in enumerate(list(openKeySet))}
    openInvMapping = {key+1:val for key,val in enumerate(list(openKeySet))}
    #
    open_mapping_path = os.path.join(sys.argv[5] +'/open_label_mapping.txt')

    with open(open_mapping_path, "wb") as out:
        json.dump(openInvMapping, out,indent=4)

    closedKeySet = set(closed_train_property_labels)
    closedMapping = {val:key+1 for key,val in enumerate(list(closedKeySet))}
    closedInvMapping = {key+1:val for key,val in enumerate(list(closedKeySet))}

    closed_mapping_path = os.path.join(sys.argv[5] +'/closed_label_mapping.txt')

    with open(closed_mapping_path, "wb") as out:
        json.dump(closedInvMapping, out,indent=4)

    genCostMatrices(pattern2regions,cost_matrices,openKeySet,closedKeySet)

    for model,dict in cost_matrices.items():
        dict['words_path'] = os.path.join(sys.argv[5]+model+"_words.dat")
        dict['words_arow_path'] = os.path.join(sys.argv[6]+model+"_words.dat")
        dict['words_ld_path'] = os.path.join(sys.argv[5]+model+"_words_ld.dat")

        dict['bigrams_path'] = os.path.join(sys.argv[5]+model+"_bigrams.dat")
        dict['bigrams_arow_path'] = os.path.join(sys.argv[6]+model+"_bigrams.dat")
        dict['bigrams_ld_path'] = os.path.join(sys.argv[5]+model+"_bigrams_ld.dat")

        dict['depgrams_path'] = os.path.join(sys.argv[5]+model+"_depgrams.dat")
        dict['depgrams_arow_path'] = os.path.join(sys.argv[6]+model+"_depgrams.dat")
        dict['depgrams_ld_path'] = os.path.join(sys.argv[5]+model+"_depgrams_ld.dat")

        dict['wordgrams_path'] = os.path.join(sys.argv[5]+model+"_wordgrams.dat")
        dict['wordgrams_arow_path'] = os.path.join(sys.argv[6]+model+"_wordgrams.dat")
        dict['wordgrams_ld_path'] = os.path.join(sys.argv[5]+model+"_wordgrams_ld.dat")

    # This has to be done after we gain the mappings
    # openle = preprocessing.LabelEncoder()
    # closedle = preprocessing.LabelEncoder()
    #
    # openle.fit(train_property_labels)

    for i,value in enumerate(train_property_labels):
        train_property_labels[i]=openMapping[value]

    # train_property_labels = openle.transform(train_property_labels)
    # print "Single training labels are ",train_property_labels

    for i,value in enumerate(closed_train_property_labels):
        closed_train_property_labels[i]=closedMapping[value]

    # closedle.fit(closed_train_property_labels)
    # closed_train_property_labels = closedle.transform(closed_train_property_labels)
    # print "Single training labels are ",closed_train_property_labels

    for model,arr in cost_matrices.items():
        if not model.startswith("single"):
            if str(model).split('_')[0]=="open":
                for open_dict in arr['cost_matrix']:
                    # print "Open dict is", open_dict
                    for key in open_dict.keys():
                        # print "Key is",key
                        open_dict[openMapping[key]] = open_dict.pop(key)
            else:
                for inner_dict in arr['cost_matrix']:
                    # print "Closed dict is", inner_dict
                    # print "Model is ",model
                    for key in inner_dict.keys():
                        # print "Key is",key
                        inner_dict[closedMapping[key]] = inner_dict.pop(key)
    '''
    Generate different training files
    '''
    generateDatFiles(cost_matrices,train_property_labels,closed_train_property_labels,train_wordlist,train_gramlist,train_bigram_list,train_wordbigram_list)

    '''
    Generate different versions of the test files.
    '''

    # Generate test files
    arow_testfile = open(os.path.join(sys.argv[6]+"words_test.dat"), 'w')
    vowpal_testfile = open(os.path.join(sys.argv[5]+"words_test.dat"), 'w')
    vowpal_open_ld_testfile = open(os.path.join(sys.argv[5]+"words_ldf_open_test.dat"), 'w')
    vowpal_closed_ld_testfile = open(os.path.join(sys.argv[5]+"words_ldf_closed_test.dat"), 'w')

    arow_testfile_bigrams = open(os.path.join(sys.argv[6]+"bigrams_test.dat"), 'w')
    vowpal_testfile_bigrams = open(os.path.join(sys.argv[5]+"bigrams_test.dat"), 'w')
    vowpal_open_ld_testfile_bigrams = open(os.path.join(sys.argv[5]+"bigrams_ldf_open_test.dat"), 'w')
    vowpal_closed_ld_testfile_bigrams = open(os.path.join(sys.argv[5]+"bigrams_ldf_closed_test.dat"), 'w')

    arow_testfile_depgrams = open(os.path.join(sys.argv[6]+"depgrams_test.dat"), 'w')
    vowpal_testfile_depgrams = open(os.path.join(sys.argv[5]+"depgrams_test.dat"), 'w')
    vowpal_open_ld_testfile_depgrams = open(os.path.join(sys.argv[5]+"depgrams_ldf_open_test.dat"), 'w')
    vowpal_closed_ld_testfile_depgrams = open(os.path.join(sys.argv[5]+"depgrams_ldf_closed_test.dat"), 'w')

    arow_testfile_wordgrams = open(os.path.join(sys.argv[6]+"wordgrams_test.dat"), 'w')
    vowpal_testfile_wordgrams = open(os.path.join(sys.argv[5]+"wordgrams_test.dat"), 'w')
    vowpal_open_ld_testfile_wordgrams = open(os.path.join(sys.argv[5]+"wordgrams_ldf_open_test.dat"), 'w')
    vowpal_closed_ld_testfile_wordgrams = open(os.path.join(sys.argv[5]+"wordgrams_ldf_closed_test.dat"), 'w')


    for i,(features,bigrams,depgrams,wordgrams) in enumerate(zip(test_wordlist,test_gramlist,test_bigram_list,test_wordbigram_list)):
        vowpal_line = str(i) + "| " + features
        arow_line = features

        vowpal_line_bigrams = str(i) + "| " + bigrams.encode('utf-8')
        arow_line_bigrams = bigrams.encode('utf-8')

        vowpal_line_depgrams = str(i) + "| " + depgrams.encode('utf-8')
        arow_line_depgrams = depgrams.encode('utf-8')

        vowpal_line_wordgrams = str(i) + "| " + wordgrams.encode('utf-8')
        arow_line_wordgrams = wordgrams.encode('utf-8')

        # Now we write the test files
        vowpal_testfile.write(vowpal_line+"\n")
        arow_testfile.write(arow_line+"\n")

        vowpal_testfile_bigrams.write(vowpal_line_bigrams+"\n")
        arow_testfile_bigrams.write(arow_line_bigrams+"\n")

        vowpal_testfile_depgrams.write(vowpal_line_depgrams+"\n")
        arow_testfile_depgrams.write(arow_line_depgrams+"\n")

        vowpal_testfile_wordgrams.write(vowpal_line_wordgrams+"\n")
        arow_testfile_wordgrams.write(arow_line_wordgrams+"\n")

        # This is to do with the latent dirichlet version
        vowpal_ldopen_line = "shared | " + features + "\n"
        vowpal_ldclosed_line = "shared | " + features + "\n"

        vowpal_ldopen_line_bigrams = "shared | " + bigrams.encode('utf-8') + "\n"
        vowpal_ldclosed_line_bigrams = "shared | " + bigrams.encode('utf-8') + "\n"

        vowpal_ldopen_line_depgrams = "shared | " + depgrams.encode('utf-8') + "\n"
        vowpal_ldclosed_line_depgrams = "shared | " + depgrams.encode('utf-8') + "\n"

        vowpal_ldopen_line_wordgrams = "shared | " + wordgrams.encode('utf-8') + "\n"
        vowpal_ldclosed_line_wordgrams = "shared | " + wordgrams.encode('utf-8') + "\n"

        for i,key in enumerate(openInvMapping.keys()):
            vowpal_ldopen_line+=str(key) + " | " + "label"+str(key) + "\n"
            vowpal_ldopen_line_bigrams+=str(key) + " | " + "label"+str(key) + "\n"
            vowpal_ldopen_line_depgrams+=str(key) + " | " + "label"+str(key) + "\n"
            vowpal_ldopen_line_wordgrams+=str(key) + " | " + "label"+str(key) + "\n"
        for i,key in enumerate(closedInvMapping.keys()):
            vowpal_ldclosed_line+=str(key) + " | " + "label"+str(key) + "\n"
            vowpal_ldclosed_line_bigrams+=str(key) + " | " + "label"+str(key) + "\n"
            vowpal_ldclosed_line_depgrams+=str(key) + " | " + "label"+str(key) + "\n"
            vowpal_ldclosed_line_wordgrams+=str(key) + " | " + "label"+str(key) + "\n"

        vowpal_open_ld_testfile.write(vowpal_ldopen_line+"\n")
        vowpal_closed_ld_testfile.write(vowpal_ldclosed_line+"\n")

        vowpal_open_ld_testfile_bigrams.write(vowpal_ldopen_line_bigrams+"\n")
        vowpal_closed_ld_testfile_bigrams.write(vowpal_ldclosed_line_bigrams+"\n")

        vowpal_open_ld_testfile_depgrams.write(vowpal_ldopen_line_depgrams+"\n")
        vowpal_closed_ld_testfile_depgrams.write(vowpal_ldclosed_line_depgrams+"\n")

        vowpal_open_ld_testfile_wordgrams.write(vowpal_ldopen_line_wordgrams+"\n")
        vowpal_closed_ld_testfile_wordgrams.write(vowpal_ldclosed_line_wordgrams+"\n")

