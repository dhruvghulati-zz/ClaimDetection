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


threshold = float(sys.argv[4])
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

property2region2value = loadMatrix(sys.argv[1])
sentencesDiscarded = 0
negativeInstances = 0
positiveInstances = 0

def absError(numer,denom):
    return abs(numer-denom)/numpy.abs(float(denom))

def findMatch(target, country):
    # print "Country for match is ", country
    return min(country, key= lambda x: absError(target, country.get(x)))

def update(sentence):
    global sentencesDiscarded
    global negativeInstances
    global positiveInstances
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
        # res = sentence.copy()
        if error<threshold:
            res.update({'predictedRegion': matchedProperty, 'meanAbsError': error})
            positiveInstances += 1
        else:
            res.update({'predictedRegion': "no_region", 'meanAbsError': 1})
            negativeInstances += 1
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

    with open(sys.argv[2]) as sentenceFile:
        sentence2locations2values = json.loads(sentenceFile.read())

    print "sentences to predict properties for:", len(sentence2locations2values['sentences'])
    '''TODO this should be able to take a MAPE threshold as argument
    '''
    predictionSentences = []
    # Note this can be made smaller for iteration purposes we don't need to use all sentences
    for sentence in sentence2locations2values['sentences'][:50000]:
        update(sentence)
#     threshold = sys.argv[4]
# pr  int "MAPE threshold is", threshold
#     print updated

    print "Sentences discarded", sentencesDiscarded
    print "Negative instances",negativeInstances
    print "Positive instances", positiveInstances

    # updated = balanceNegativeExamples(updated)

    # predictRegion(sentence2locations2values,property2region2value)
    # The new sentence file with predicted statistical property

    outputFile = sys.argv[3]
    with open(sys.argv[3], "wb") as out:
        json.dump(predictionSentences, out,indent=4)

    # properties = json.loads(open(os.path.dirname(os.path.abspath(sys.argv[1])) + "/featuresKept.json").read())