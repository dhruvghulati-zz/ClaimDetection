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


# TODO - need to make sure I can account for there not being some sentences.

predictionSentences = []

# import sys; print(sys.executable)
#
# import os; print(os.getcwd())
#
# import sys; print(sys.path)

'''
TODO - this should never predict properties that are not in our list - we should not load in the whole KB
'''

def loadMatrix(jsonFile):
    print "loading from file " + jsonFile
    with open(jsonFile) as freebaseFile:
        country2property2value = json.loads(freebaseFile.read())

    regions = set([])
    valueCounter = 0
    for country, property2value in country2property2value.items():
        # Check for nan values and remove them
        # print "Country is ",country
        # print "Property2Value is ",property2value
        for property, value in property2value.items():
            # Deleting any properties we don't care about'
            # print "these are the properties:",properties
            # print "this is the property to check", property
            # if property not in properties:
            #     del property2value[property]
                # print property2value
                # print "REMOVED:", "property:",property, "value:", value
            if not np.isfinite(value):
                del property2value[property]
                print "REMOVED:", value, " for ", property, " ", country
        if len(property2value) == 0:
            del country2property2value[country]
            # print "REMOVED property:", country, " no values left"
        else:
            valueCounter += len(property2value)
            regions = regions.union(set(property2value.keys()))

    print len(country2property2value), " properties"
    print len(regions),  " unique regions"
    print valueCounter, " values loaded"
    return country2property2value


def absError(numer,denom):
    return abs(denom-numer)/np.abs(float(numer))

def findMatch(target, country):
    filtered_country = {property: country.get(property) for property in properties}
    filtered_country = {k: v for k, v in filtered_country.items() if v}
    openMatch = min(country, key= lambda x: absError(target, country.get(x)))
    closedMatch = min(filtered_country, key= lambda x: absError(target, filtered_country.get(x)))
    return openMatch, closedMatch

def update(sentence):
    global negativeOpenThresholdInstances
    global positiveOpenThresholdInstances
    global negativeClosedThresholdInstances
    global positiveClosedThresholdInstances
    global threshold
    # print 'Checking sentence: ', sentence
    (c,target), = sentence.get("region-val_pair").items()
    # print "Checking country: ", c,"and value: ", target
    # res = sentence.copy()
    # print property2region2value
    if c in property2region2value:
        res = sentence.copy()
        country = property2region2value[c]
        matchedProperty, closedMatch = findMatch(target,country)
        error = absError(target, country.get(matchedProperty))
        closedError = absError(target, country.get(closedMatch))
        res.update({'predictedPropertyClosed': closedMatch, 'closedMeanAbsError': closedError})
        res.update({'predictedPropertyOpen': matchedProperty, 'meanAbsError': error})
        if error<threshold:
            res.update({'predictedPropertyOpenThreshold': matchedProperty, 'meanAbsError': error})
            positiveOpenThresholdInstances += 1
        else:
            res.update({'predictedPropertyOpenThreshold': "no_region", 'meanAbsError': error})
            negativeOpenThresholdInstances +=1
        if closedError<threshold:
            res.update({'predictedPropertyClosedThreshold': closedMatch, 'closedMeanAbsError': closedError})
            positiveClosedThresholdInstances +=1
        else:
            res.update({'predictedPropertyClosedThreshold': "no_region", 'closedMeanAbsError': error})
            negativeClosedThresholdInstances +=1
        predictionSentences.append(res)

def mape_threshold_region_predictor(testSentences):

    for sentence in testSentences:
        update(sentence)
    return predictionSentences

def testSentenceLabels(dict_list):
    temp_properties =[]
    for i, property in enumerate(properties):
        temp_properties.append(property.split("/")[3])
        # TODO - Command line issue as I had hard coded the location of these files ../../, in command line remove
    print "Temporary properties are",len(temp_properties)
    for subdir, dirs, files in os.walk('data/labeled_claims'):
        # This is causing errors
        for file in files:
            # print os.path.join(subdir, file)
            filepath = subdir + os.sep + file
            if filepath.endswith(".xlsx"):
                # print "Filepath is",filepath
                wb = xlrd.open_workbook(filepath, encoding_override="utf-8")
                for s in wb.sheets():
                    # read header values into the list
                    keys = [s.cell(0, col_index).value for col_index in xrange(s.ncols)]
                    mape_index = keys.index("mape")
                    claim_index = keys.index("label")
                    kbval_index = keys.index("kb_value")
                    value_index = keys.index("extracted_value")
                    alias_index = keys.index("alias")
                    dep_index = keys.index("dep")
                    sentence_index = keys.index("sentence")
                    country_index = keys.index("country")
                    for row_index in xrange(1, s.nrows):
                        sentence = {}
                        sentence['parsedSentence'] = {}
                        sentence['property'] = {}
                        sentence['mape'] = s.cell(row_index, mape_index).value
                        sentence['claim'] = s.cell(row_index, claim_index).value
                        sentence['dependencies'] = s.cell(row_index, dep_index).value
                        # TODO - on command line this should change to remove ../..
                        if filepath=="data/labeled_claims/internet_users_percent_population_claims.xlsx" or filepath=="data/labeled_claims/population_growth_rate_claims.xlsx":
                            # print filepath
                            # print "False"
                            extracted_country = s.cell(row_index, country_index).value
                            # print "Country is ",extracted_country
                            sentence_noSlots = s.cell(row_index, sentence_index).value
                            # print "Sentence is ",sentence_noSlots
                            extracted_value = s.cell(row_index, value_index).value
                            # print "Value is ",extracted_value
                            # sentence_noSlots.replace(str(extracted_country),"<location>empty</location>")
                            # sentence_noSlots.replace(str(extracted_value),"<number>empty</number>")
                            finalSentence = sentence_noSlots.replace(str(extracted_country),"<location>empty</location>").replace(str(extracted_value),"<number>empty</number>")
                            # sentenceTokens = sentence_noSlots.split()
                            # for i,token in enumerate(sentenceTokens):
                            #     if token==extracted_country:
                            #         sentenceTokens[i]="<location>empty</location>"
                            #     if token==extracted_value:
                            #         sentenceTokens[i]="<number>empty</number>"
                            # sentence_noSlots = (" ").join(sentenceTokens)
                            # print "New sentence is",finalSentence
                            sentence['parsedSentence'] = finalSentence
                        else:
                            for col_index in xrange(s.ncols):
                                if isinstance(s.cell(row_index,col_index).value, unicode):
                                    # print "True"
                                        if s.cell(row_index,col_index).value.find(r"<location>")>-1 or \
                                                    s.cell(row_index,col_index).value.find(r"<number>")>-1:
                                            sentence['parsedSentence'] = s.cell(row_index,col_index).value

                        if sentence['claim'] =="Y":
                            sentence['claim'] = 1
                        # Note some sentences have a question mark as a claim
                        # else:
                        #     sentence['claim'] = 0
                        elif sentence['claim']=="N":
                            sentence['claim'] = 0
                        sentence['kb_value'] = s.cell(row_index, kbval_index).value
                        # sentence['value'] = s.cell(row_index, mape_index).value
                        sentence['region-val_pair'] = {s.cell(row_index, alias_index).value: s.cell(row_index, value_index).value}
                        # dict_list = [dict() for x in xrange(1, s.nrows)]
                        # d = {keys[mape_index]: s.cell(row_index, mape_index).value}
                        # dict_list.append(d)
                        for col_index in xrange(s.ncols):
                            if isinstance(s.cell(row_index,col_index).value, unicode):
                                if s.cell(row_index,col_index).value in temp_properties:
                                    sentence['property'] = "/location/statistical_region/" + s.cell(row_index,col_index).value
                                        # print "true"
                        dict_list.append(sentence)

    # print dict_list
    return dict_list


def labelSlotFiltering(testLabels):
    global threshold
    print "MAPE threshold is ",threshold,"\n"
    # This is to make the slot format same as training
    for i, dataTriples in enumerate(testLabels):
        # print "Old sentence is" ,dataTriples['parsedSentence']
        if dataTriples['parsedSentence']:
            # slotText = re.sub(r'(?s)(<location>)(.*?)(</location>)', r"LOCATION_SLOT", dataTriples['parsedSentence'])
            slotText = re.sub(r'(<location>.*</location>)', r"LOCATION_SLOT", dataTriples['parsedSentence'])
            slotText = re.sub(r'(<number>.*</number>)', r"NUMBER_SLOT", slotText)
            # print "New sentence is ", slotText
            dataTriples['parsedSentence'] = slotText
    # print "Total test labels is", len(testLabels)

    # Now we give the MAPEs labels:

    for i, dataTriples in enumerate(testLabels):
        # print "Old sentence is" ,dataTriples['parsedSentence']
        dataTriples['mape_label'] = {}
        # if dataTriples['mape']:
        if dataTriples['mape']<threshold:
            dataTriples['mape_label']=1
        else:
            dataTriples['mape_label']=0
            # print "New sentence is ", slotText
    # print "Total test labels is", len(testLabels)
    # print "Total positive labels is ",len([dataTriples['mape_label'] for a,dataTriples in enumerate(testLabels) if dataTriples['mape_label']==1])
    # print "Total negative labels is ",len([dataTriples['mape_label'] for a,dataTriples in enumerate(testLabels) if dataTriples['mape_label']==0])
    # print "Total test labels with parsed sentences is ",len([dataTriples['parsedSentence'] for a,dataTriples in enumerate(testLabels) if dataTriples['parsedSentence']!={}])
    # print "Total test labels with no mape is ",len([dataTriples['mape'] for a,dataTriples in enumerate(testLabels) if dataTriples['mape']=={}])
    # print "Total test labels with no property is ",len([dataTriples['property'] for a,dataTriples in enumerate(testLabels) if dataTriples['property']=={}])
    # Here I give "no region" to any sentence not a claim
    for i, dataTriples in enumerate(testLabels):
        # if dataTriples['claim']==0 and dataTriples['claim'] is not None:
        #     # print "Claim is ", dataTriples['property']
        #     dataTriples['property']="no_region"
    # Finally, specify the threshold that was used
        dataTriples['threshold']=threshold
    return testLabels

# `python src/main/testFeatures.py data/featuresKept.json data/output/testLabels.json data/output/hyperTestLabels.json $var data/freebaseTriples.json data/output/devLabels.json`
if __name__ == "__main__":

    # np.seterr(all='print')
    rng = np.random.RandomState(101)

    # properties = json.loads(open(os.path.dirname(sys.argv[1]).read()))
    with open(sys.argv[1]) as featuresKept:
        properties = json.loads(featuresKept.read())
    print "We have ",len(properties),"features kept\n"

    threshold = float(sys.argv[4])

    property2region2value = loadMatrix(sys.argv[5])

    # This step is necessary for getting data from xlsx
    dict_list = []

    testLabels = testSentenceLabels(dict_list)

    print "Extracted test labels with parsed sentences from Excel sheets is", len(testLabels)



    testLabels = labelSlotFiltering(testLabels)

    # print "Total positive labels is ",len([dataTriples['mape_label'] for a,dataTriples in enumerate(testLabels) if dataTriples['mape_label']==1])
    # print "Total negative labels is ",len([dataTriples['mape_label'] for a,dataTriples in enumerate(testLabels) if dataTriples['mape_label']==0])

    # print "Total test labels with no mape is ",len([dataTriples['mape'] for a,dataTriples in enumerate(testLabels) if dataTriples['mape']=={}])

    print "Total test labels with property is",len([dataTriples['property'] for a,dataTriples in enumerate(testLabels) if dataTriples['property']!={}])

    # print "Total test labels with parsed sentences is ",len([dataTriples['parsedSentence'] for a,dataTriples in enumerate(testLabels) if dataTriples['parsedSentence']!={}])

    cleanTestLabels = []

    # Here we remove blanks and clean up the test set - note we ignore some properties because we are not sure if they contain a claim or not - ?
    for i, dataTriples in enumerate(testLabels):
        if dataTriples['mape']!={} and dataTriples['parsedSentence']!={} and dataTriples['property']!={} and dataTriples['claim']!="?" and dataTriples['claim']!="":
            cleanTestLabels.append(dataTriples)

    print "Total clean test labels with no ? claim is", len(cleanTestLabels),"\n"

    properties.append("no_region")

    # Here I add the same predictor I used for the training data on the test data
    # I need the full properties again to do a prediciton

    negativeOpenThresholdInstances = 0
    positiveOpenThresholdInstances = 0
    negativeClosedThresholdInstances = 0
    positiveClosedThresholdInstances = 0

    cleanTestLabels = mape_threshold_region_predictor(cleanTestLabels)

    print "Negative open threshold instances",negativeOpenThresholdInstances
    print "Positive open threshold instances", positiveOpenThresholdInstances
    print "Negative closed threshold instances",negativeClosedThresholdInstances
    print "Positive closed threshold instances", positiveClosedThresholdInstances

    finalTestLabels = []
    devLabels = []
    hyperTestLabels = []
    rng.shuffle(cleanTestLabels)

    propertiesCovered=[]
    '''
    TODO - make sure all statistical regions covered in the hyper test labels
    '''
    s = set(dataTriples['property'] for i, dataTriples in enumerate(cleanTestLabels))

    print "Here are the unique properties in the test labels\n"
    for x in s:
        print x
    print "There are ", len(s), " properties\n"

    print "Here are the unique properties in the features kept\n"
    for x in properties:
        print x
    print "There are ", len(properties), " properties\n"

    for i, dataTriples in enumerate(cleanTestLabels):
        if i<500:
            devLabels.append(dataTriples)
        if i>=500 and i<1500:
            hyperTestLabels.append(dataTriples)
        if i>=1500 and i<len(cleanTestLabels):
            finalTestLabels.append(dataTriples)
    print "Number of hyper sentences is", len(hyperTestLabels),"\n"

    print "Total positive mape labels in hyperLabels is ",len([dataTriples['mape_label'] for a,dataTriples in enumerate(hyperTestLabels) if dataTriples['mape_label']==1])
    print "Total negative mape labels in hyperLabels  is ",len([dataTriples['mape_label'] for a,dataTriples in enumerate(hyperTestLabels) if dataTriples['mape_label']==0])
    print "Total positive claim labels in hyperLabels is ",len([dataTriples['claim'] for a,dataTriples in enumerate(hyperTestLabels) if dataTriples['claim']==1])
    print "Total negative claim labels in hyperLabels is ",len([dataTriples['claim'] for a,dataTriples in enumerate(hyperTestLabels) if dataTriples['claim']==0])
    # print "Total unique properties in hyperLabels with no property is ",len([dataTriples['property'] for a,dataTriples in enumerate(hyperTestLabels) if dataTriples['property']=={}])

    uniquePropHyper = set(dataTriples['property'] for a,dataTriples in enumerate(hyperTestLabels))
    #
    # for x in s:
    #     print x
    print "There are ",len(uniquePropHyper), "unique properties covered in hyperTestLabels","\n"

    print "Number of dev sentences is", len(devLabels)

    print "Total positive mape labels in devLabels is ",len([dataTriples['mape_label'] for a,dataTriples in enumerate(devLabels) if dataTriples['mape_label']==1])
    print "Total negative mape labels in devLabels  is ",len([dataTriples['mape_label'] for a,dataTriples in enumerate(devLabels) if dataTriples['mape_label']==0])
    print "Total positive claim labels in devLabels is ",len([dataTriples['claim'] for a,dataTriples in enumerate(devLabels) if dataTriples['claim']==1])
    print "Total negative claim labels in devLabels is ",len([dataTriples['claim'] for a,dataTriples in enumerate(devLabels) if dataTriples['claim']==0])
    # print "Total unique properties in devLabels with no property is ",len([dataTriples['property'] for a,dataTriples in enumerate(devLabels) if dataTriples['property']=={}])

    uniquePropDev = set(dataTriples['property'] for a,dataTriples in enumerate(devLabels))
    #
    # for x in s:
    #     print x
    print "There are ",len(uniquePropDev), "unique properties covered in devLabels","\n"


    with open(sys.argv[2], "wb") as out:
            #Links the sentences to the region-value pairs
            json.dump(finalTestLabels, out,indent=4)

    with open(sys.argv[3], "wb") as out:
            #Links the sentences to the region-value pairs
            json.dump(hyperTestLabels, out,indent=4)

    with open(sys.argv[6], "wb") as out:
            #Links the sentences to the region-value pairs
            json.dump(devLabels, out,indent=4)