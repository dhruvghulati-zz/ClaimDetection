'''

Then it trains a model using a matrix of text patterns and database

It then obtains a ranking of the patterns

And the checks files parsed and NER'ed JSONs by Stanford CoreNLP and produces the following structure:

Location:[dep1:[val1, val2], dep1:[val1, val2, ...]]

It produces a ranking of the sentences according to the relation at question and scores each value by MAPE

python src/main/factChecker.py data/freebaseTriples.json data/mainMatrixFiltered.json population 0.03125 data/output/testLabels.json data/locationNames data/aliases.json out/claims_identified/population.csv

'''

import json
import sys
import os
import glob
import codecs
import operator
import buildMatrix
import baselinePredictor



# training data
# load the FreeBase file
with open(sys.argv[1]) as freebaseFile:
    region2property2value = json.loads(freebaseFile.read())

# we need to make it property2region2value
property2region2value = {}
for region, property2value in region2property2value.items():
    for property, value in property2value.items():
        if property not in property2region2value:
            property2region2value[property] = {}
        property2region2value[property][region] = value

# text patterns
textMatrix = baselinePredictor.BaselinePredictor.loadMatrix(sys.argv[2])

print "Number of filtered patterns used for training", len(textMatrix)

# specify which ones are needed:
property = "/location/statistical_region/" + sys.argv[3]

# first let's train a model

predictor = baselinePredictor.BaselinePredictor()
params = [True, float(sys.argv[4])]

# train
predictor.trainRelation(property, property2region2value[property], textMatrix, sys.stdout, params)

print "patterns kept for this property:"
print predictor.property2patterns[property].keys()


# # parsed texts to check
# parsedJSONDir = sys.argv[5]
#
# # get all the files
# jsonFiles = glob.glob(parsedJSONDir + "/*.json")
#
#
# print str(len(jsonFiles)) + " files to process"

# load the hardcoded names
tokenizedLocationNames = []
names = codecs.open(sys.argv[6], encoding='utf-8').readlines()
for name in names:
    tokenizedLocationNames.append(unicode(name).split())
print "Dictionary with hardcoded tokenized location names"
print tokenizedLocationNames

# get the aliases
# load the file
with open(sys.argv[7]) as jsonFile:
    region2aliases = json.loads(jsonFile.read())

# so we first need to take the location2aliases dict and turn in into aliases to region
alias2region = {}
for region, aliases in region2aliases.items():
    # add the location as alias to itself
    alias2region[region] = region
    for alias in aliases:
        # so if this alias is used for a different location
        if alias in alias2region and region!=alias2region[alias]:
            alias2region[alias] = None
            alias2region[alias.lower()] = None
        else:
            # remember to add the lower
            alias2region[alias] = region
            alias2region[alias.lower()] = region

# now filter out the Nones
for alias, region in alias2region.items():
    if region == None:
        print "alias ", alias, " ambiguous"
        del alias2region[alias]

print alias2region

# store the result: sentence, country, number, nearestPattern, euclidDistance, correctNumber, MAPE

tsv = open(sys.argv[8], "wb")

headers = ['claim','prediction','sentence', 'region', 'kb_region', 'property','labeled_property', 'kb_value', 'mape_support_scaling_param', 'pattern', 'value', 'MAPE']

tsv.write("\t".join(headers) + "\n")

'''
Now we apply to test data.
'''

with codecs.open(sys.argv[5]) as testFile:
    parsedSentences = json.loads(testFile.read())

for i, sentenceDict in enumerate(parsedSentences):
    patternsApplied = []
    for pattern in sentenceDict['patterns']:
        # print "Pattern being tested is",pattern
        if pattern in predictor.property2patterns[property].keys():
            patternsApplied.append(pattern)

    if len(patternsApplied) > 0:

        sentenceText = sentenceDict['parsedSentence']
        location = sentenceDict['location-value-pair'].keys()[0]
        number = sentenceDict['location-value-pair'].values()[0]
        claim = sentenceDict['claim']
        prediction = 1
        labeled_property = sentenceDict['property'].split('/')[3]

        print "Sentence: " + sentenceText.encode('utf-8')
        print "location in text " + location.encode('utf-8') + " is known as " + region.encode('utf-8') + " in FB with known " + property + " value " + str(property2region2value[property][region])
        print "confidence level= " + str(len(patternsApplied)) + "\t" + str(patternsApplied)
        print "sentence states that " + location.encode('utf-8') + " has " + property + " value " + str(number)
        if property2region2value[property][region] != 0.0:
            mape = abs(number - property2region2value[property][region]) / float(abs(property2region2value[property][region]))
            print "MAPE: " + str(mape)
        else:
            print "MAPE undefined"
            mape = "undef"
        print "------------------------------"
        details = [str(claim).encode('utf-8'),str(prediction).encode('utf-8'),sentenceText.encode('utf-8'), location.encode('utf-8'), region.encode('utf-8'), sys.argv[3], labeled_property.encode('utf-8'),str(property2region2value[property][region]), str(len(patternsApplied)),str(patternsApplied), str(number), str(mape)]
        tsv.write("\t".join(details) + "\n")

tsv.close()
