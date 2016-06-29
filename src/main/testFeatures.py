'''
This labels each sentence in the training excel files as being 1 or 0 based on if their MAPE is above 0.05 or not
'''

import os
import xlrd
import json
import sys
import re
from random import shuffle
import numpy as np

rng = np.random.RandomState(101)

# python testFeatures.py data/featuresKept.json data/output/testLabels.json data/output/hyperTestLabels.json 0.10

# properties = json.loads(open(os.path.dirname(sys.argv[1]).read()))


threshold = float(sys.argv[4])

with open(sys.argv[1]) as testSentences:
        properties = json.loads(testSentences.read())
        for i, property in enumerate(properties):
            properties[i] = property.split("/")[3]
            # print property[i].split("/")[1]
print "We have ",len(properties),"features kept"

'''
TODO put the sentences in test json in same format as training i.e. location slot, number slot as features, and also hard label the sentences according to mapes > 5% per our training concept.
'''

def testSentenceLabels():
    for subdir, dirs, files in os.walk('../../data/labeled_claims'):
        dict_list = []
        for file in files:
            # print os.path.join(subdir, file)
            filepath = subdir + os.sep + file

            if filepath.endswith(".xlsx"):
                wb = xlrd.open_workbook(filepath)

                for s in wb.sheets():
                    # read header values into the list
                    keys = [s.cell(0, col_index).value for col_index in xrange(s.ncols)]
                    mape_index = keys.index("mape")
                    claim_index = keys.index("label")
                    kbval_index = keys.index("kb_value")
                    value_index = keys.index("extracted_value")
                    alias_index = keys.index("alias")
                    # keys = ['Bla']
                    # for dict in dict_list:
                    for row_index in xrange(1, s.nrows):
                        sentence = {}
                        sentence['parsedSentence'] = {}
                        sentence['property'] = {}
                        sentence['mape'] = s.cell(row_index, mape_index).value
                        sentence['claim'] = s.cell(row_index, claim_index).value
                        if sentence['claim'] =="Y":
                            sentence['claim'] = 1
                        else:
                            sentence['claim'] = 0
                        sentence['kb_value'] = s.cell(row_index, kbval_index).value
                        # sentence['value'] = s.cell(row_index, mape_index).value
                        sentence['region-val_pair'] = {s.cell(row_index, alias_index).value: s.cell(row_index, value_index).value}
                        # dict_list = [dict() for x in xrange(1, s.nrows)]
                        # d = {keys[mape_index]: s.cell(row_index, mape_index).value}
                        # dict_list.append(d)

                        for col_index in xrange(s.ncols):
                            if isinstance(s.cell(row_index,col_index).value, unicode):
                                if s.cell(row_index,col_index).value.find("<location>")>-1 or s.cell(row_index,col_index).value.find("<number>")>-1:
                                    sentence['parsedSentence'] = s.cell(row_index,col_index).value
                                if s.cell(row_index,col_index).value in properties:
                                    sentence['property'] = "location/statistical_region/" + s.cell(row_index,col_index).value
                                    # print "true"
                        dict_list.append(sentence)

    # print dict_list
    return dict_list


def labelSlotFiltering(testLabels):
    global threshold
    print threshold
    # This is to make the slot format same as training
    for i, dataTriples in enumerate(testLabels):
        # print "Old sentence is" ,dataTriples['parsedSentence']
        if dataTriples['parsedSentence']:
            slotText = re.sub(r'(?s)(<location>)(.*?)(</location>)', r"LOCATION_SLOT", dataTriples['parsedSentence'])
            slotText = re.sub(r'(?s)(<number>)(.*?)(</number>)', r"NUMBER_SLOT", slotText)
            # print "New sentence is ", slotText
            dataTriples['parsedSentence'] = slotText
    print "Total test labels is", len(testLabels)
    print "Total test labels with parsed sentences is ",len([dataTriples['parsedSentence'] for a,dataTriples in enumerate(testLabels) if dataTriples['parsedSentence']!={}])
    print "Total test labels with no mape is ",len([dataTriples['mape'] for a,dataTriples in enumerate(testLabels) if dataTriples['mape']=={}])
    print "Total test labels with no property is ",len([dataTriples['property'] for a,dataTriples in enumerate(testLabels) if dataTriples['property']=={}])

    # Now we give the MAPEs labels:

    for i, dataTriples in enumerate(testLabels):
        # print "Old sentence is" ,dataTriples['parsedSentence']
        dataTriples['mape_label'] = {}
        if dataTriples['mape']:
            if dataTriples['mape']>threshold:
                dataTriples['mape_label']=0
            else:
                dataTriples['mape_label']=1
            # print "New sentence is ", slotText
    print "Total test labels is", len(testLabels)
    print "Total positive labels is ",len([dataTriples['mape_label'] for a,dataTriples in enumerate(testLabels) if dataTriples['mape_label']==1])
    print "Total negative labels is ",len([dataTriples['mape_label'] for a,dataTriples in enumerate(testLabels) if dataTriples['mape_label']==0])

    for i, dataTriples in enumerate(testLabels):
        if dataTriples['claim']==0 & dataTriples['claim']:
            # print "Claim is ", dataTriples['property']
            dataTriples['property']="no_region"

    return testLabels

testLabels = testSentenceLabels()
testLabels = labelSlotFiltering(testLabels)

cleanTestLabels = []

# Here we remove blanks
for i, dataTriples in enumerate(testLabels):
    if dataTriples['mape']!={} and dataTriples['parsedSentence']!={} and dataTriples['property']!={}:
        cleanTestLabels.append(dataTriples)

finalTestLabels = []
hyperTestLabels = []
# cleanTestLabels = [dataTriples for i,dataTriples in enumerate(cleanTestLabels)]
rng.shuffle(cleanTestLabels)
# shuffle(cleanTestLabels)
# print cleanTestLabels
#
# for i in cleanTestLabels:
#     print i

propertiesCovered=[]

'''
TODO - make sure all statistical regions covered in the hyper test labels
'''
for i, dataTriples in enumerate(cleanTestLabels):
    # if len(propertiesCovered)!=len(properties)+1:
    #     if dataTriples['property'] not in propertiesCovered:
    #         propertiesCovered.append(dataTriples['property'])
    #         print len(propertiesCovered)
    #         hyperTestLabels.append(dataTriples)
    if i<1000:
        hyperTestLabels.append(dataTriples)
    if i>=1000 and i<len(cleanTestLabels):
        finalTestLabels.append(dataTriples)
# print "Number of hyper sentences is", len(hyperTestLabels)




with open(sys.argv[2], "wb") as out:
        #Links the sentences to the region-value pairs
        json.dump(finalTestLabels, out,indent=4)

with open(sys.argv[3], "wb") as out:
        #Links the sentences to the region-value pairs
        json.dump(hyperTestLabels, out,indent=4)
