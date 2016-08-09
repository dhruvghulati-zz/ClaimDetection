'''
This is a file that predicts the closest statistical property for each region-value pair for each sentence from Freebase

I use a MAPE method to calculate.
'''
from __future__ import division

import operator
import numpy as np
import sys
import json

'''
TODO - evaluate the information extraction via 4 fold training?
How to test coverage and average MAPE per statistical property?
'''

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




def absError(numer,denom):
    # print "Denominator is ",denom
    # print "Numerator is",numer
    # print "Error is ",abs(denom-numer)/np.abs(float(numer)),"\n"
    return abs(denom-numer)/np.abs(float(numer))

def findMatch(target, country):
    # print "Property-value pairs for match is ", country
    # print "Closed properties are", properties
    filtered_country = {property: country.get(property) for property in properties}
    # Remove all values with no value for the property we want e.g. Qatar
    filtered_country = {k: v for k, v in filtered_country.items() if v}
    # print "Filtered country is",filtered_country
    openCostVector = sorted([float(absError(target, v)) for k, v in country.items()])
    closedCostVector = sorted([float(absError(target, v)) for k, v in filtered_country.items()])
    #
    # print "Open Cost array is",openCostVector
    # print "Closed Cost array is",closedCostVector

    openCostDict = {k: absError(target, v) for k, v in country.items() if v}
    # closedCostDict = sorted(closedCostDict.items(), key=operator.itemgetter(1))

    closedCostDict = {k: absError(target, v) for k, v in filtered_country.items() if v}
    # openCostDict = sorted(openCostDict.items(), key=operator.itemgetter(1))

    # print "Open Cost dict is",closedCostDict
    # print "Closed Cost dict is",openCostDict


    openMatch = min(country, key= lambda x: float(absError(target, country.get(x))))
    closedMatch = min(filtered_country, key= lambda x: float(absError(target, filtered_country.get(x))))
    # print "Open match is ",openMatch
    # print "Closed match is ",closedMatch

    return openMatch, closedMatch,openCostVector,closedCostVector,openCostDict,closedCostDict

def update(sentence):

    global sentencesDiscarded
    global negativeOpenThresholdInstances
    global positiveOpenThresholdInstances
    global negativeOpenInstances
    global positiveOpenInstances
    global negativeClosedThresholdInstances
    global positiveClosedThresholdInstances
    global negativeClosedInstances
    global positiveClosedInstances
    global threshold
    # print 'Checking sentence: ', sentence
    (c,target), = sentence.get("location-value-pair").items()
    # print "Checking country: ", c,"and value: ", target
    # res = sentence.copy()
    # print property2region2value
    if c in property2region2value:
        res = sentence.copy()
        country = property2region2value[c]
        # This is the open matched property
        matchedProperty, closedMatch, openCostArr, closedCostArr,openDict, closedDict = findMatch(target,country)
        # print "This is the openCostArr",openCostArr
        # print "This is the closedCostArr",closedCostArr
        # print "This is matched property: ", matchedProperty
        error = float(absError(target, country.get(matchedProperty)))
        closedError = float(absError(target, country.get(closedMatch)))
        # print "Closed error is",closedError
        # print "Open error is",error

        res.update({'predictedPropertyClosed': closedMatch, 'closedMeanAbsError': closedError,
                    'closedCostArr':closedCostArr,'closedCostDict':closedDict

                    })
        res.update({'predictedPropertyOpen': matchedProperty, 'meanAbsError': error,
                    'openCostArr':openCostArr,'openCostDict':openDict
                    })
        if error<threshold:
            res.update({'predictedPropertyOpenThreshold': matchedProperty})
            positiveOpenThresholdInstances += 1
        else:
            res.update({'predictedPropertyOpenThreshold': "no_region"})
            negativeOpenThresholdInstances +=1
        if closedError<threshold:
            res.update({'predictedPropertyClosedThreshold': closedMatch})
            positiveClosedThresholdInstances +=1
        else:
            res.update({'predictedPropertyClosedThreshold': "no_region"})
            negativeClosedThresholdInstances +=1
        predictionSentences.append(res)
    else:
        # print c," is not in the Freebase country list"
        sentencesDiscarded+=1
        # del sentence
    # return res


# def balanceNegativeExamples(sentencePred):



# src/main/propertyPredictor.py data/freebaseTriples.json data/sentenceMatrixFiltered.json data/output/predictedProperties.json 0.05

if __name__ == "__main__":

    threshold = float(sys.argv[4])

    with open(sys.argv[2]) as sentenceFile:
        sentence2locations2values = json.loads(sentenceFile.read())

    with open(sys.argv[5]) as featuresKept:
        properties = json.loads(featuresKept.read())
    print "We have ",len(properties),"features kept\n"

    print "sentences to predict properties for:", len(sentence2locations2values['sentences'])
    '''TODO this should be able to take a MAPE threshold as argument
    '''
    predictionSentences = []

    property2region2value = loadMatrix(sys.argv[1])

    sentencesDiscarded = 0
    negativeOpenThresholdInstances = 0
    positiveOpenThresholdInstances = 0
    # negativeOpenInstances = 0
    # positiveOpenInstances = 0
    negativeClosedThresholdInstances = 0
    positiveClosedThresholdInstances = 0
    # negativeClosedInstances = 0
    # positiveClosedInstances = 0

    # Note this can be made smaller for iteration purposes we don't need to use all sentences
    for sentence in sentence2locations2values['sentences'][:50000]:
        update(sentence)
    #     threshold = sys.argv[4]
    # pr  int "MAPE threshold is", threshold
    #     print updated

    print "Sentences discarded", sentencesDiscarded
    print "Sentences with matching countries", len(predictionSentences)
    # print "Negative open instances",negativeOpenInstances
    # print "Positive open instances", positiveOpenInstances
    print "Negative open threshold instances",negativeOpenThresholdInstances
    print "Positive open threshold instances", positiveOpenThresholdInstances
    # print "Negative closed instances",negativeClosedInstances
    # print "Positive closed instances", positiveClosedInstances
    print "Negative closed threshold instances",negativeClosedThresholdInstances
    print "Positive closed threshold instances", positiveClosedThresholdInstances


    # updated = balanceNegativeExamples(updated)

    # predictRegion(sentence2locations2values,property2region2value)
    # The new sentence file with predicted statistical property

    outputFile = sys.argv[3]

    # print predictionSentences

    with open(sys.argv[3], "wb") as out:
        json.dump(predictionSentences, out,indent=4)

    # properties = json.loads(open(os.path.dirname(os.path.abspath(sys.argv[1])) + "/featuresKept.json").read())