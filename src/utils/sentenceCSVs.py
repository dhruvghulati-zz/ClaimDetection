'''
/Users/dhruv/Documents/university/ClaimDetection/data/output/sentenceSlotsFiltered.json
/Users/dhruv/Documents/university/ClaimDetection/data/sentenceLabels.csv
/Users/dhruv/Documents/university/ClaimDetection/data/mainProperties.csv
/Users/dhruv/Documents/university/ClaimDetection/data/featuresKept.json
/Users/dhruv/Documents/university/ClaimDetection/data/remainingProperties.csv
/Users/dhruv/Documents/university/ClaimDetection/data/propertiesOfInterestClean.json
'''

import sys
import json
import csv
import numpy as np

rng = np.random.RandomState(101)

with open(sys.argv[1]) as sentenceFile:
    sentence2locations2values = json.loads(sentenceFile.read())

rng.shuffle(sentence2locations2values)

with open(sys.argv[4]) as featureFile:
    properties = json.loads(featureFile.read())

with open(sys.argv[6]) as fullPropFile:
    fullProperties = json.loads(fullPropFile.read())

with open(sys.argv[2], 'w') as csvfile:
    fieldnames = ['sentence',
                  'Which of the statistics is this sentence talking about, if any?',
                  'Is the sentence a claim, no_claim or n/a?'
                  ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    # [50000:60000]
    for sentence in sentence2locations2values[50000:60000]:
        sentence = unicode(sentence['parsedSentence']).encode("utf-8")
        # print sentence['parsedSentence']
        writer.writerow({'sentence': sentence})

with open(sys.argv[3], 'w') as csvfile:
    fieldnames = ['statistic']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for property in fullProperties:
        property = unicode(property).encode("utf-8").split("/")[3]
        # print sentence['parsedSentence']
        if property not in ['rent50_0','rent50_1','rent50_2','rent50_3','rent50_4']:
            writer.writerow({'statistic': property})
    for property in properties:
        if property not in fullProperties:
            property = unicode(property).encode("utf-8").split("/")[3]
            writer.writerow({'statistic': property})

# with open(sys.argv[5], 'w') as csvfile:
#     fieldnames = ['statistic']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#     writer.writeheader()
#     for i,sentence in enumerate(fullProperties):
#         property = unicode(sentence).encode("utf-8").split("/")[3]
#     # print sentence['parsedSentence']
#         if property not in properties:
#             writer.writerow({'statistic': property})

