'''
This is a file that predicts the closest statistical property for each region-value pair for each sentence from Freebase

I use a MAPE method to calculate.
'''

import operator
import numpy
import sys
import json

'''
TODO - evaluate the information extraction via 4 fold training?
How to test coverage and average MAPE per statistical property?
Do we constrain at this step which properties you are predicting?
'''



# Check if OK to constrain properties being predicted
# with open(sys.argv[4]) as featuresKept:
#         properties = json.loads(featuresKept.read())
#         properties.append("no_region")

def MAPE(predDict, trueDict, verbose=False):
    absPercentageErrors = {}
    keysInCommon = list(set(predDict.keys()) & set(trueDict.keys()))

    # print keysInCommon
    for key in keysInCommon:
        # avoid 0's
        if trueDict[key] != 0:
            absError = abs(predDict[key] - trueDict[key])
            absPercentageErrors[key] = absError/numpy.abs(trueDict[key])

    if len(absPercentageErrors) > 0:
        if verbose:
            print "MAPE results"
            sortedAbsPercentageErrors = sorted(absPercentageErrors.items(), key=operator.itemgetter(1))
            print "top-5 predictions"
            print "region:pred:true"
            for idx in xrange(5):
                print sortedAbsPercentageErrors[idx][0].encode('utf-8'), ":", predDict[sortedAbsPercentageErrors[idx][0]], ":", trueDict[sortedAbsPercentageErrors[idx][0]]
            print "bottom-5 predictions"
            for idx in xrange(5):
                print sortedAbsPercentageErrors[-idx-1][0].encode('utf-8'), ":", predDict[sortedAbsPercentageErrors[-idx-1][0]], ":", trueDict[sortedAbsPercentageErrors[-idx-1][0]]

        return numpy.mean(absPercentageErrors.values())
    else:
        return float("inf")

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
            if not numpy.isfinite(value):
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
    return abs(numer-denom)/numpy.abs(float(denom))

def findMatch(target, country):
    # print "Country for match is ", country
    return min(country, key= lambda x: absError(target, country.get(x)))

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
        matchedProperty = findMatch(target,country)
        # print "This is matched property: ", matchedProperty
        error = absError(target, country.get(matchedProperty))
        if matchedProperty in properties:
            res.update({'predictedPropertyOpen': matchedProperty, 'meanAbsError': error})
            res.update({'predictedPropertyClosed': matchedProperty, 'meanAbsError': error})
            positiveOpenInstances +=1
            positiveClosedInstances +=1
            if error<threshold:
                res.update({'predictedPropertyClosedThreshold': matchedProperty, 'meanAbsError': error})
                res.update({'predictedPropertyOpenThreshold': matchedProperty, 'meanAbsError': error})
                positiveOpenThresholdInstances += 1
            else:
                res.update({'predictedPropertyClosedThreshold': "no_region", 'meanAbsError': error})
                res.update({'predictedPropertyOpenThreshold': "no_region", 'meanAbsError': error})
                negativeOpenThresholdInstances +=1
        else:
            res.update({'predictedPropertyClosed': "unmatched_region", 'meanAbsError': error})
            res.update({'predictedPropertyClosedThreshold': "unmatched_region", 'meanAbsError': error})
            negativeClosedInstances+=1
            positiveOpenInstances +=1
            if error<threshold:
                res.update({'predictedPropertyOpen': matchedProperty, 'meanAbsError': error})
                res.update({'predictedPropertyOpenThreshold': matchedProperty, 'meanAbsError': error})
                positiveOpenThresholdInstances += 1
                positiveClosedThresholdInstances +=1
            else:
                res.update({'predictedPropertyOpen': matchedProperty, 'meanAbsError': error})
                res.update({'predictedPropertyOpenThreshold': "no_region", 'meanAbsError': error})
                negativeOpenThresholdInstances +=1
                negativeClosedThresholdInstances +=1
        # res = sentence.copy()

        predictionSentences.append(res)
        # return res
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
    negativeOpenInstances = 0
    positiveOpenInstances = 0
    negativeClosedThresholdInstances = 0
    positiveClosedThresholdInstances = 0
    negativeClosedInstances = 0
    positiveClosedInstances = 0

    # Note this can be made smaller for iteration purposes we don't need to use all sentences
    for sentence in sentence2locations2values['sentences'][:50000]:
        update(sentence)
    #     threshold = sys.argv[4]
    # pr  int "MAPE threshold is", threshold
    #     print updated

    print "Sentences discarded", sentencesDiscarded
    print "Sentences with matching countries", len(predictionSentences)
    print "Negative open instances",negativeOpenInstances
    print "Positive open instances", positiveOpenInstances
    print "Negative open threshold instances",negativeOpenThresholdInstances
    print "Positive open threshold instances", positiveOpenThresholdInstances
    print "Negative closed instances",negativeClosedInstances
    print "Positive closed instances", positiveClosedInstances
    print "Negative closed threshold instances",negativeClosedThresholdInstances
    print "Positive closed threshold instances", positiveClosedThresholdInstances


    # updated = balanceNegativeExamples(updated)

    # predictRegion(sentence2locations2values,property2region2value)
    # The new sentence file with predicted statistical property

    outputFile = sys.argv[3]
    with open(sys.argv[3], "wb") as out:
        json.dump(predictionSentences, out,indent=4)

    # properties = json.loads(open(os.path.dirname(os.path.abspath(sys.argv[1])) + "/featuresKept.json").read())