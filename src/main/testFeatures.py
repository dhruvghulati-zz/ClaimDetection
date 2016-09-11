'''

Extracts the sentences and their key metadata and labels from the annotated Excel files produced from a previous model.

Splits the data into dev, hypertest, and final test labels, as well as full labels.

This labels each sentence in the training excel files as being 1 or 0 based on if their MAPE is above 0.05 or not

python src/main/testFeatures.py data/featuresKept.json 0.05 data/freebaseTriples.json data/output/devLabels.json data/output/testLabels.json data/output/fullTestLabels.json data/output/cleanFullLabels.json data/output/cleanFullLabelsDistant.json

'''

import os
import xlrd
import json
import sys
import re
import numpy as np
import ast

def loadMatrix(jsonFile):
    print "loading from file " + jsonFile
    with open(jsonFile) as freebaseFile:
        country2property2value = json.loads(freebaseFile.read())

    regions = set([])
    valueCounter = 0
    for country, property2value in country2property2value.items():
        # Note, in this case we do not delete any properties, we keep them open
        for property, value in property2value.items():
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
    print len(regions), " unique regions"
    print valueCounter, " values loaded"
    return country2property2value


def absError(numer, denom):
    return np.abs(denom - numer) / np.abs(float(numer))


def findMatch(target, country):
    filtered_country = {property: country.get(property) for property in properties}
    filtered_country = {k: v for k, v in filtered_country.items() if v}
    openMatch = min(country, key=lambda x: absError(target, country.get(x)))
    closedMatch = min(filtered_country, key=lambda x: absError(target, filtered_country.get(x)))
    return openMatch, closedMatch


def update(sentence,threshold):

    # negativeOpenThresholdInstances = 0
    # positiveOpenThresholdInstances = 0
    # negativeClosedThresholdInstances = 0
    # positiveClosedThresholdInstances = 0

    (c, target), = sentence.get("location-value-pair").items()
    res = sentence.copy()
    if c in property2region2value:
        country = property2region2value[c]
        matchedProperty, closedMatch = findMatch(target, country)
        error = absError(target, country.get(matchedProperty))
        closedError = absError(target, country.get(closedMatch))
        res.update({'predictedPropertyClosed': closedMatch, 'closedMeanAbsError': closedError})
        res.update({'predictedPropertyOpen': matchedProperty, 'meanAbsError': error})
        if error < threshold:
            res.update({'predictedPropertyOpenThreshold': matchedProperty, 'meanAbsError': error})
            # positiveOpenThresholdInstances += 1
        else:
            res.update({'predictedPropertyOpenThreshold': "no_region", 'meanAbsError': error})
            # negativeOpenThresholdInstances += 1
        if closedError < threshold:
            res.update({'predictedPropertyClosedThreshold': closedMatch, 'closedMeanAbsError': closedError})
            # positiveClosedThresholdInstances += 1
        else:
            res.update({'predictedPropertyClosedThreshold': "no_region", 'closedMeanAbsError': error})
            # negativeClosedThresholdInstances += 1

    # print "Negative open threshold instances", negativeOpenThresholdInstances
    # print "Positive open threshold instances", positiveOpenThresholdInstances
    # print "Negative closed threshold instances", negativeClosedThresholdInstances
    # print "Positive closed threshold instances", positiveClosedThresholdInstances

    return res

def mape_threshold_region_predictor(testSentences,threshold):

    predictionSentences = []
    for i,sentence in enumerate(testSentences):
        predictionSentences.append(update(sentence,threshold))
    # Here I give "no region" to any sentence not a claim
    for i, dataTriples in enumerate(predictionSentences):
        dataTriples['threshold'] = threshold
    return predictionSentences


def testSentenceLabels(input_properties):

    dict_list = []
    temp_properties = []
    for i, property in enumerate(input_properties):
        temp_properties.append(property.split("/")[3])
        # TODO - Command line issue as I had hard coded the location of these files ../../, in command line remove
    print "Temporary properties are", len(temp_properties)
    for subdir, dirs, files in os.walk('data/labeled_claims'):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".xlsx"):
                print "Filepath is",filepath
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
                        lines = str(s.cell(row_index, dep_index).value)
                        try:
                           sentence['patterns'] = ast.literal_eval(lines)
                        except (ValueError,SyntaxError):
                            continue
                        text = str(s.cell(row_index, dep_index).value)
                        text = re.sub(r'\[u|\]', "", text)
                        text = text.split(",")[0]
                        text = re.sub(r'\'', "", text)
                        text = text.split("+")
                        bigrams = [text[i:i + 2] for i in xrange(len(text) - 2)]
                        bigrams = [("+").join(bigram).encode('utf-8') for bigram in bigrams]
                        bigrams = (' ').join(map(str, bigrams))
                        sentence['depPath'] = bigrams
                        # TODO - on command line this should change to remove ../..
                        if filepath == "data/labeled_claims/internet_users_percent_population_claims.xlsx" or filepath == "data/labeled_claims/population_growth_rate_claims.xlsx":
                            extracted_country = s.cell(row_index, country_index).value
                            # print "Country is ",extracted_country
                            sentence_noSlots = s.cell(row_index, sentence_index).value
                            # print "Sentence is ",sentence_noSlots
                            extracted_value = s.cell(row_index, value_index).value
                            finalSentence = sentence_noSlots.replace(str(extracted_country),
                                                                     "<location>empty</location>").replace(
                                str(extracted_value), "<number>empty</number>")
                            sentence['parsedSentence'] = finalSentence
                        else:
                            for col_index in xrange(s.ncols):
                                if isinstance(s.cell(row_index, col_index).value, unicode):
                                    # print "True"
                                    if s.cell(row_index, col_index).value.find(r"<location>") > -1 or \
                                                    s.cell(row_index, col_index).value.find(r"<number>") > -1:
                                        sentence['parsedSentence'] = s.cell(row_index, col_index).value

                        if sentence['claim'] == "Y":
                            sentence['claim'] = 1
                        # Note some sentences have a question mark as a claim
                        # else:
                        #     sentence['claim'] = 0
                        elif sentence['claim'] == "N":
                            sentence['claim'] = 0
                        sentence['kb_value'] = s.cell(row_index, kbval_index).value
                        # sentence['value'] = s.cell(row_index, mape_index).value
                        sentence['location-value-pair'] = {
                            s.cell(row_index, alias_index).value: s.cell(row_index, value_index).value}
                        for col_index in xrange(s.ncols):
                            if isinstance(s.cell(row_index, col_index).value, unicode):
                                if s.cell(row_index, col_index).value in temp_properties:
                                    sentence['property'] = "/location/statistical_region/" + s.cell(row_index,
                                                                                                    col_index).value
                                    # print "true"
                        dict_list.append(sentence)

    # print dict_list
    return dict_list


def labelSlotFiltering(testLabels):
    global threshold
    print "MAPE threshold is ", threshold, "\n"
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
        # print "Old sentence is" ,dataTriples['parsedSentence']
        dataTriples['mape_label'] = {}
        dataTriples['categorical_mape_label'] = {}
        # if dataTriples['mape']:
        if dataTriples['mape'] < threshold:
            dataTriples['mape_label'] = 1
            dataTriples['categorical_mape_label'] = dataTriples['property']
        else:
            dataTriples['mape_label'] = 0
            dataTriples['categorical_mape_label'] = "no_region"
    # Now we give the MAPEs labels

    return testLabels


if __name__ == "__main__":

    # np.seterr(all='print')
    rng = np.random.RandomState(101)

    with open(sys.argv[1]) as featuresKept:
        properties = json.loads(featuresKept.read())
    print "We have ", len(properties), "features kept\n"

    with open(sys.argv[7]) as newTestLabels:
        newTestLabels = json.loads(newTestLabels.read())
    print "We have ", len(newTestLabels), "fresh test labels\n"

    threshold = float(sys.argv[2])

    property2region2value = loadMatrix(sys.argv[3])

    # This step is necessary for getting data from xlsx
    testLabels = testSentenceLabels(properties)

    print "Extracted test labels with parsed sentences from Excel sheets is", len(testLabels)

    testLabels = labelSlotFiltering(testLabels)

    cleanTestLabels1 = []

    for i, dataTriples in enumerate(testLabels):
        if dataTriples['property'] != {} and dataTriples['property'] != "":
            cleanTestLabels1.append(dataTriples)

    print "Total test labels with property is", len(cleanTestLabels1)

    cleanTestLabels2 = []

    for i, dataTriples in enumerate(cleanTestLabels1):
        if dataTriples['parsedSentence'] != {} and dataTriples['parsedSentence'] != "":
            cleanTestLabels2.append(dataTriples)

    print "Total test labels with parsed sentence is", len(cleanTestLabels2)

    cleanTestLabels3 = []

    for i, dataTriples in enumerate(cleanTestLabels2):
        if dataTriples['mape'] != {} and dataTriples['mape'] != "":
            cleanTestLabels3.append(dataTriples)

    print "Total test labels with mape is", len(cleanTestLabels3)

    cleanTestLabels = []

    # Here we remove blanks and clean up the test set - note we ignore some properties because we are not sure if they contain a claim or not - ?
    for i, dataTriples in enumerate(cleanTestLabels3):
        if dataTriples['claim'] != "?" and dataTriples['claim'] != "":
            cleanTestLabels.append(dataTriples)

    print "Total clean test labels with claim labels is", len(cleanTestLabels), "\n"

    properties.append("no_region")

    # Here I add the same predictor I used for the training data on the test data
    # I need the full properties again to do a prediciton

    cleanTestLabels = mape_threshold_region_predictor(cleanTestLabels,threshold)

    newTestLabels = mape_threshold_region_predictor(newTestLabels,threshold)


    finalTestLabels = []
    devLabels = []
    hyperTestLabels = []
    rng.shuffle(cleanTestLabels)

    propertiesCovered = []
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
        if i < 1500:
            devLabels.append(dataTriples)
        # if i >= 500 and i < 1500:
        #     hyperTestLabels.append(dataTriples)
        if i >= 1500 and i < len(cleanTestLabels):
            finalTestLabels.append(dataTriples)
    # print "Number of hyper sentences is", len(hyperTestLabels), "\n"
    #
    # print "Total positive mape labels in hyperLabels is ", len(
    #     [dataTriples['mape_label'] for a, dataTriples in enumerate(hyperTestLabels) if dataTriples['mape_label'] == 1])
    # print "Total negative mape labels in hyperLabels  is ", len(
    #     [dataTriples['mape_label'] for a, dataTriples in enumerate(hyperTestLabels) if dataTriples['mape_label'] == 0])
    # print "Total positive claim labels in hyperLabels is ", len(
    #     [dataTriples['claim'] for a, dataTriples in enumerate(hyperTestLabels) if dataTriples['claim'] == 1])
    # print "Total negative claim labels in hyperLabels is ", len(
    #     [dataTriples['claim'] for a, dataTriples in enumerate(hyperTestLabels) if dataTriples['claim'] == 0])
    # print "Total unique properties in hyperLabels with no property is ", len(
    #     [dataTriples['property'] for a, dataTriples in enumerate(hyperTestLabels) if dataTriples['property'] == {}])
    #
    # uniquePropHyper = set(dataTriples['property'] for a, dataTriples in enumerate(hyperTestLabels))
    # #
    # # for x in s:
    # #     print x
    # print "There are ", len(uniquePropHyper), "unique properties covered in hyperTestLabels", "\n"

    print "Number of dev sentences is", len(devLabels)

    # print "Total positive mape labels in devLabels is ", len(
    #     [dataTriples['mape_label'] for a, dataTriples in enumerate(devLabels) if dataTriples['mape_label'] == 1])
    # print "Total negative mape labels in devLabels  is ", len(
    #     [dataTriples['mape_label'] for a, dataTriples in enumerate(devLabels) if dataTriples['mape_label'] == 0])
    # print "Total positive claim labels in devLabels is ", len(
    #     [dataTriples['claim'] for a, dataTriples in enumerate(devLabels) if dataTriples['claim'] == 1])
    # print "Total negative claim labels in devLabels is ", len(
    #     [dataTriples['claim'] for a, dataTriples in enumerate(devLabels) if dataTriples['claim'] == 0])

    uniquePropDev = set(dataTriples['property'] for a, dataTriples in enumerate(devLabels))
    print "There are ", len(uniquePropDev), "unique properties covered in devLabels", "\n"

    with open(sys.argv[4], "wb") as out:
        # Links the sentences to the region-value pairs
        json.dump(devLabels, out, indent=4)

    with open(sys.argv[5], "wb") as out:
        # Links the sentences to the region-value pairs
        json.dump(finalTestLabels, out, indent=4)

    with open(sys.argv[6], "wb") as out:
        # Links the sentences to the region-value pairs
        json.dump(cleanTestLabels, out, indent=4)

    with open(sys.argv[8], "wb") as out:
        # Links the sentences to the region-value pairs
        json.dump(newTestLabels, out, indent=4)
