'''
This takes in 2 files from the sentence slots which
 1. Is for training (includes all sentences)
2. is for labelling (discarded overly dense sentences)

- It optionally can output files with sentences with 0 values (from sentence) removed
- It applies alias locations to both files

It removes:

1. Duplicate original sentence and location-value-pairs for the training sentences (which can be caused due to the LOCATION_SLOT LOCATION_SLOT issue with countries like Hong Kong
2. Duplicate parsed sentences overall for the labelled sentences.

Ensures no potential training sentences could every appear in the predefined test sentences.

Outputs:

-Training sentences (all of them)
-Unique sentences for labelling, any of which can be sent out for labelling.

Arguments:

/Users/dhruv/Documents/university/ClaimDetection/data/output/sentenceRegionValue.json
/Users/dhruv/Documents/university/ClaimDetection/data/output/sentenceSlotsFiltered.json
/Users/dhruv/Documents/university/ClaimDetection/data/aliases.json
/Users/dhruv/Documents/university/ClaimDetection/data/output/fullTestLabels.json
/Users/dhruv/Documents/university/ClaimDetection/data/output/cleanFullLabels.json
/Users/dhruv/Documents/university/ClaimDetection/data/sentenceMatrixFilteredZero.json
/Users/dhruv/Documents/university/ClaimDetection/data/output/uniqueSentenceLabels.json
'''

import json
import numpy as np
import sys
import itertools
import operator


def removeZeros(inputDict):
    # Now remove any sentences with values of 0:
    finalpattern2locations2values = {}
    finalpattern2locations2values['sentences'] = []
    for i, sentence in enumerate(inputDict['sentences']):
        for key, value in sentence['location-value-pair'].iteritems():
            if value != 0.0:
                finalpattern2locations2values['sentences'].append(sentence)
    print "Unique sentences after deleting 0 values:", len(finalpattern2locations2values['sentences'])
    return finalpattern2locations2values


def convertAliases(region2aliases):
    # so we first need to take the location2aliases dict and turn in into aliases to region
    alias2region = {}
    for region, aliases in region2aliases.items():
        # add the location as alias to itself
        alias2region[region] = region
        for alias in aliases:
            # so if this alias is used for a different location
            if alias in alias2region and region != alias2region[alias]:
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
    return alias2region


def applyAliases(aliases, sentenceDicts):
    for index, dataTriples in enumerate(sentenceDicts):
        for location, value in dataTriples['location-value-pair'].items():
            region = location
            if location in aliases:
                region = aliases[location]
            elif location.lower() in aliases:
                region = aliases[location.lower()]
            dataTriples['location-value-pair'] = {region: value}


# helps detect errors
np.seterr(all='raise')

# load the file
print "Loading the model file...\n"
with open(sys.argv[1]) as jsonFile:
    pattern2locations2values = json.loads(jsonFile.read())

print "Loading the slots file...\n"
with open(sys.argv[2]) as sentenceSlotsFull:
    fullSentenceSlots = json.loads(sentenceSlotsFull.read())

print "Model sentences before filtering:", len(pattern2locations2values['sentences'])
print "labelling sentences before filtering:", len(fullSentenceSlots)

modelVals = operator.itemgetter(u"sentence", u"location-value-pair")
slotVals = operator.itemgetter(u"parsedSentence")

pattern2locations2values['sentences'].sort(key=modelVals)
fullSentenceSlots.sort(key=slotVals)

result = []
for k, g in itertools.groupby(pattern2locations2values['sentences'], modelVals):
    result.append(g.next())
pattern2locations2values['sentences'][:] = result

result = []
for k, g in itertools.groupby(fullSentenceSlots, slotVals):
    result.append(g.next())
fullSentenceSlots[:] = result

print "Unique sentences after filtering:", len(pattern2locations2values['sentences'])
print "Unique labelling sentences after filtering:", len(fullSentenceSlots)

# load the file
print "Loading the aliases file...\n"
with open(sys.argv[3]) as jsonFile:
    aliasFile = json.loads(jsonFile.read())

alias2region = convertAliases(aliasFile)

# ok, let's traverse now all the patterns and any locations we find we match them case independently to the aliases and replace them with the location

print "Applying aliases for model...\n"

applyAliases(alias2region, pattern2locations2values['sentences'])

print "Applying aliases for manual labelling...\n"

applyAliases(alias2region, fullSentenceSlots)

print "Ensuring no sentences from this pool could occur anywhere in my training set...\n"

# load the file
print "Loading the old test file...\n"
with open(sys.argv[4]) as oldTestFile:
    oldTestFile = json.loads(oldTestFile.read())

print "Loading the new test file...\n"
with open(sys.argv[5]) as newTestFile:
    newTestFile = json.loads(newTestFile.read())

print "Old test files:", len(oldTestFile)
print "New test files:", len(newTestFile)

uniqueTestSentences = []
oldTestFile.extend(newTestFile)
for myDict in oldTestFile:
    if myDict not in uniqueTestSentences:
        uniqueTestSentences.append(myDict)

st = {(tuple(d["location-value-pair"].items()), d["parsedSentence"]) for d in uniqueTestSentences}

pattern2locations2valuesUnique = {'sentences': []}

pattern2locations2valuesUnique['sentences'][:] = (d for d in pattern2locations2values['sentences'] if (
tuple(d["location-value-pair"].items()), d["parsedSentence"]) not in st)

print "Training pool sentences after removing test sentences:", len(pattern2locations2valuesUnique['sentences'])

print "Writing to filtered file for model...\n"

with open(sys.argv[6], "wb") as out:
    json.dump(pattern2locations2valuesUnique, out, indent=4, encoding='utf-8')

print "Writing to filtered file for manual labelling\n"

with open(sys.argv[7], "wb") as out:
    # Cant do ensure ascii false
    json.dump(fullSentenceSlots, out, indent=4, encoding='utf-8')
