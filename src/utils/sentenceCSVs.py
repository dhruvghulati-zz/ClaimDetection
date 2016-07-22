import sys
import json
import csv
import numpy as np

rng = np.random.RandomState(101)

with open(sys.argv[1]) as sentenceFile:
    sentence2locations2values = json.loads(sentenceFile.read())

sentence2locations2values = sentence2locations2values['sentences']
rng.shuffle(sentence2locations2values)
# sentence2locations2values = OrderedDict(items)

# print sentence2locations2values[:10]

# sentence2locations2values = sorted(sentence2locations2values.items(), key=lambda x: random.random())

with open(sys.argv[4]) as featureFile:
    properties = json.loads(featureFile.read())

with open(sys.argv[6]) as fullPropFile:
    fullProperties = json.loads(fullPropFile.read())

with open(sys.argv[2], 'w') as csvfile:
    fieldnames = ['sentence',
                  'Which one of the 16 statistics is this sentence talking about, if any?',
                  'Which one of the remaining 54 statistics is this sentence talking about, if not talking about the set of 16? If it is none of the 54, please put "no_property',
                  'Is the sentence subjective or objective?',
                  'Is the sentence an affirmative or negation?',
                  'Is the sentence a statement or question?',
                  'Is the sentence factual or counterfactual?',
                  'Is the sentence a statement about the past, present, or future?',
                  'Is the sentence in plain language or gibberish?'
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
    for property in properties:
        property = unicode(property).encode("utf-8").split("/")[3]
        # print sentence['parsedSentence']
        writer.writerow({'statistic': property})

with open(sys.argv[5], 'w') as csvfile:
    fieldnames = ['statistic']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i,sentence in enumerate(fullProperties):
        property = unicode(sentence).encode("utf-8").split("/")[3]
    # print sentence['parsedSentence']
        if property not in properties:
            writer.writerow({'statistic': property})

