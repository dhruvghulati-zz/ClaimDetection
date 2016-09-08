'''
This takes out the final claim /test labels and produces a dev and test file. It also augments the claim labels with the additional metadata that was lost during the labelling process where a lot of unnecessary metadata was removed but will be useful for evaluating distant supervision metrics.

/Users/dhruv/Documents/university/ClaimDetection/data/clean_labels/claim_labels.csv
/Users/dhruv/Documents/university/ClaimDetection/data/output/cleanDevLabels.json
/Users/dhruv/Documents/university/ClaimDetection/data/output/cleanTestLabels.json
/Users/dhruv/Documents/university/ClaimDetection/data/output/cleanFullLabels.json
/Users/dhruv/Documents/university/ClaimDetection/data/output/uniqueSentenceLabelsDep.json

'''

import csv
import json
import sys
import numpy as np
import codecs
import copy

rng = np.random.RandomState(101)

csvfile = open(sys.argv[1], 'r')

fieldnames = ("parsedSentence","claim","property")
reader = csv.DictReader(csvfile, fieldnames)
# Skip the header
next(reader)

testArray = []

for i,row in enumerate(reader):
    row['property'] = "/location/statistical_region/"+row['property']
    if row['claim']=="claim": row['claim']=1
    else: row['claim']=0
    row['parsedSentence'] = row['parsedSentence'].decode("utf-8")
    testArray.append(row)

# print testArray

rng.shuffle(testArray)

# Now append information from uniqueSentenceLabels including dependency paths.
print "processing " + sys.argv[5]
with open(sys.argv[5]) as jsonFile:
    parsedSentences = json.loads(jsonFile.read())

for i,triples in enumerate(parsedSentences):
    for j,labelTriples in enumerate(testArray):
        if triples['parsedSentence'] ==labelTriples['parsedSentence']:
            labelTriples.update(triples)

print testArray[0]

# Now adapt the bigrams so it matches the dev set:
for i, labelTriples in enumerate(testArray):
    if 'depPath' in labelTriples.keys():
        bigrams = copy.deepcopy(labelTriples['depPath'])
        bigrams = [("+").join(bigram).encode('utf-8') for bigram in bigrams]
        bigrams = (' ').join(map(str, bigrams))
        labelTriples['depPath'] = bigrams


# Now split into dev and test
testLabels = []
devLabels = []

for i, dataTriples in enumerate(testArray):
    if i<250:
        devLabels.append(dataTriples)
    if i>=250 and i<len(testArray):
        testLabels.append(dataTriples)

print "Number of dev sentences is", len(devLabels),"\n"
print "Number of test sentences is", len(testLabels),"\n"

# Generate some statistics
posTest = 0
posDev = 0

for i, row in enumerate(testLabels):
    if row['property'].split("/")[3] =="statistic_not_listed" or row['property'].split("/")[3] =="not_about_economic_statistics" and row['claim']==1:
        posTest+=1

for i, row in enumerate(devLabels):
    if row['property'].split("/")[3] =="statistic_not_listed" or row['property'].split("/")[3] =="not_about_economic_statistics" and row['claim']==1:
        posDev+=1

print "Number of +ve dev sentences is", posDev,"\n"
print "Number of +ve test sentences is", posTest,"\n"


with open(sys.argv[2], "wb") as out:
    json.dump(devLabels, out,indent=4)

with open(sys.argv[3], "wb") as out:
    json.dump(testLabels, out,indent=4)

with open(sys.argv[4], "wb") as out:
    json.dump(testArray, out,indent=4)