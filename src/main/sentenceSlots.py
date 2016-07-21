import json
import sys
import glob
import networkx
import re
import copy
import numpy
import codecs
import itertools
from nltk.text import Text
'''
TODO -
'''
# this function performs a dictNER matching to help with names that Stanford NER fails
# use with caution, it ignores everything apart from the tokens, over-writing existing NER tags
def dictLocationMatching(sentence, tokenizedLocations):
    # first re-construct the sentence as a string
    wordsInSentence = []
    for token in sentence["tokens"]:
        wordsInSentence.append(token["word"])
    # print "Words in sentence are: " , wordsInSentence
    for tokLoc in tokenizedLocations:
        # print "Words in sentence are: " , wordsInSentence
        # print "Token location is: " , tokLoc
        tokenSeqs = [(i, i+len(tokLoc)) for i in range(len(wordsInSentence)) if wordsInSentence[i:i+len(tokLoc)] == tokLoc]
        # print "Token sequence is: " , tokenSeqs
        for tokenSeq in tokenSeqs:
            for tokenNo in range(tokenSeq[0], tokenSeq[1]):
                sentence["tokens"][tokenNo]["ner"]  = "LOCATION"

def getNumbers(sentence):
    # a number can span over multiple tokens
    tokenIDs2number = {}
    for idx, token in enumerate(sentence["tokens"]):
        # avoid only tokens known to be dates or part of locations
        # This only takes actual numbers into account thus it ignores things like "one million"
        # and also treating "500 millions" as "500"
        if token["ner"] not in ["DATE", "LOCATION", "PERSON", "ORGANIZATION", "MISC"]:
            try:
                # this makes sure that 123,123,123.23 which fails the float test, becomes 123123123.23 which is good
                tokenWithoutCommas = re.sub(",([0-9][0-9][0-9])", "\g<1>", token["word"])
                number = float(tokenWithoutCommas)
                # we want this to avoid taking in nan, inf and -inf as floats
                if numpy.isfinite(number):
                    ids = [idx]
                    # check the next token if it is million or thousand
                    if len(sentence["tokens"]) > idx+1:
                        if sentence["tokens"][idx+1]["word"].startswith("trillion"):
                            number = number * 1000000000000
                            ids.append(idx+1)
                            # print "Value extracted is: ", number
                        elif sentence["tokens"][idx+1]["word"].startswith("billion"):
                            number = number * 1000000000
                            ids.append(idx+1)
                            # print "Value extracted is: ", number
                        elif sentence["tokens"][idx+1]["word"].startswith("million"):
                            number = number * 1000000
                            ids.append(idx+1)
                            # print "Tokens of sentence are: ", sentence["tokens"]
                            # print "Value extracted is: ", number
                        elif sentence["tokens"][idx+1]["word"].startswith("thousand"):
                            number = number * 1000
                            ids.append(idx+1)
                            # print "Tokens of sentence are: ", sentence["tokens"]
                            # print "Value extracted is: ", number

                    tokenIDs2number[tuple(ids)] = number

            except ValueError:
                pass
    return tokenIDs2number


def getLocations(sentence):
    # note that a location can span multiple tokens
    tokenIDs2location = {}
    currentLocation = []
    for idx, token in enumerate(sentence["tokens"]):
        # if it is a location token add it:
        if token["ner"] == "LOCATION":
            currentLocation.append(idx)
        # if it is a no location token
        else:
            # check if we have just finished a location
            if len(currentLocation) > 0:
                # convert the tokenID to a tuple (immutable) and put the name there
                locationTokens = []
                for locIdx in currentLocation:
                    locationTokens.append(sentence["tokens"][locIdx]["word"])
                    # print "Location extracted is: ", sentence["tokens"][locIdx]["word"]

                tokenIDs2location[tuple(currentLocation)] = " ".join(locationTokens)
                currentLocation = []

    return tokenIDs2location


# python src/main/sentenceSlots.py data/train_jsons data/output/sentenceRegionValue.json data/locationNames data/test_jsons data/output/testData.json data/output/sentenceSlotsFull.json
if __name__ == "__main__":

    parsedJSONDir = sys.argv[1]

    labelFile = sys.argv[6]

    discardFile = sys.argv[8]

    wordDensityThreshold = float(sys.argv[7])

    # get all the files
    jsonFiles = glob.glob(parsedJSONDir + "/*.json")

    # one json to rule them all, the sentenceRegionValue.json
    outputFile = sys.argv[2]

    # this forms the columns using the lexicalized dependency and surface patterns
    pattern2location2values = {}

    sentences2location2values = {"sentences": []}

    sentences2location2valuesSlots = []

    sentences2location2valuesDiscarded = []

    print str(len(jsonFiles)) + " files to process"

    # load the hardcoded names (if any):
    tokenizedLocationNames = []
    if len(sys.argv) > 3:
        names = codecs.open(sys.argv[3], encoding='utf-8').readlines()
        for name in names:
            # print unicode(name).split()
            tokenizedLocationNames.append(unicode(name).split())
    # print "Dictionary with hardcoded tokenized location names"
    # print tokenizedLocationNames

    # Use len(jsonFiles) for all, 100 for testing
    for jsonFileName in itertools.islice(jsonFiles , 0, 100):
    # for fileCounter, jsonFileName in enumerate(jsonFiles):
        # For each file in the HTML JSON
        print "processing " + jsonFileName
        with codecs.open(jsonFileName) as jsonFile:
            parsedSentences = json.loads(jsonFile.read())

        for sentence in parsedSentences:
            # fix the ner tags
            if len(tokenizedLocationNames)>0:
                dictLocationMatching(sentence, tokenizedLocationNames)

            tokenIDs2number = getNumbers(sentence)
            tokenIDs2location = getLocations(sentence)
            # if there was at least one location and one number build the dependency graph:
            # Check if len(sentence["tokens"])<120 step is valid
            if len(tokenIDs2number) > 0 and len(tokenIDs2location) > 0 and len(sentence["tokens"])<120:

                wordsInSentence = []

                for token in sentence["tokens"]:
                    wordsInSentence.append(token["word"])
                sample = " ".join(wordsInSentence)

                # This is the sentence.
                # print "Original sentence is : ", sample
                # for each pair of location and number
                # get the pairs of each and find their dependency paths (might be more than one)


                for locationTokenIDs, location in tokenIDs2location.items():
                    for numberTokenIDs, number in tokenIDs2number.items():
                        # print "Location token IDs are: ", locationTokenIDs
                        # print "Number token IDs are: ", numberTokenIDs
                        # print "Location is: ", location
                        # print "Number is: ", number
                        sampleTokens = sample.split()
                        sentenceDict = {}

                        sentenceDict["sentence"] = sample

                        sentenceDict["location-value-pair"] = {location:number}

                        # Check that each sentence contains at max 3 location/number slots

                        # print "Sample tokens are: ", sampleTokens
                        for numberTokenID in numberTokenIDs:
                             for locationTokenID in locationTokenIDs:
                                # TODO - adjust the tokens here
                                # print "Number token ID is : ",numberTokenID
                                # print "Location token ID is : ",locationTokenID
                                sampleTokens[locationTokenID] = "LOCATION_SLOT"
                                sampleTokens[numberTokenID] = "NUMBER_SLOT"
                                slotSentence = (" ").join(sampleTokens)
                                sentenceDict["parsedSentence"] = slotSentence
                                sentences2location2values["sentences"].append(sentenceDict)
                                sentences2location2valuesSlots.append(sentenceDict)

                # Account for millions etc. in the word density calculation
                for i,token in enumerate(sampleTokens):
                    if (token == "LOCATION_SLOT" or token == "NUMBER_SLOT") and token==prev_Word and prev_Word is not None:
                        # print "Old sample tokens",sampleTokens
                        sampleTokens.remove(sampleTokens[i-1])
                        # print "New sample tokens",sampleTokens
                        if i>0 and i<(len(sampleTokens)-1):
                            prev_Word = token
                    else:
                        if i>0 and i<(len(sampleTokens)-1):
                            prev_Word = token
                        continue

                locationCount=0
                numberCount=0

                # Remove items with too many location and number slots
                denseSentence = False
                tooManySlots = False
                # print "Here are the full tokens",slotSampleTokens
                # for i,sentence in enumerate(sentences2location2valuesSlots):
                #     sentence['parsedSentence']=(" ").join(sampleTokens)
                analyser = Text(sampleTokens)
                # Check the density of any word in sentences not being too high (likely to be too many location/number slots) on a per sentence basis

                # print slotSampleTokens

                for token in sampleTokens:
                    if token =="LOCATION_SLOT":
                        locationCount+=1
                    elif token =="NUMBER_SLOT":
                        numberCount+=1
                    # print slotSampleTokens
                    # print token
                    # print "Count is",analyser.count(token)
                    # print "Sentence length is ",len(slotSampleTokens)
                    if locationCount>3 or numberCount>3:
                    # print "Number of locations", len(tokenIDs2location)
                    # print "Number of values", len(tokenIDs2number)
                        tooManySlots = True
                        break
                    wordDensity = float(analyser.count(token))/float(len(sampleTokens))
                    # print wordDensity
                    if wordDensity>wordDensityThreshold:
                        # print "wordDensity is",wordDensity
                        # print "threshold is",wordDensityThreshold
                        denseSentence = True
                        # "Exiting loop..."
                        break

                for i,sentence in enumerate(sentences2location2valuesSlots):
                    if tooManySlots or denseSentence:
                        sentences2location2valuesDiscarded.append(sentence)
                        del sentences2location2valuesSlots[i]


    # This is about accounting for millions etc to account for bag of words

    prev_Word=None

    # TODO - need to also delete training instances with multiple region value-pairs, and do this for all countries
    for i,sentence in enumerate(sentences2location2valuesSlots):
        sampleTokens = sentence['parsedSentence'].split()
        for i,token in enumerate(sampleTokens):
            if (token == "LOCATION_SLOT" or token == "NUMBER_SLOT") and token==prev_Word and prev_Word is not None:
                # print "Old sample tokens",sampleTokens
                sampleTokens.remove(sampleTokens[i-1])
                # print "New sample tokens",sampleTokens
                if i>0 and i<(len(sampleTokens)-1):
                    prev_Word = token
            else:
                if i>0 and i<(len(sampleTokens)-1):
                    prev_Word = token
                continue
        sentence['parsedSentence']=(' ').join(sampleTokens)

    print "Model sentence length",len(sentences2location2values['sentences'])
    print "Labelled sentence length",len(sentences2location2valuesSlots)
    print "Discarded sentence length",len(sentences2location2valuesDiscarded)

    with open(outputFile, "wb") as out:
        #Links the sentences to the region-value pairs
        json.dump(sentences2location2values, out,indent=4)

    with open(labelFile, "wb") as out:
        #Links the sentences to the region-value pairs
        json.dump(sentences2location2valuesSlots, out,indent=4)

    with open(discardFile, "wb") as out:
        #Links the sentences to the region-value pairs
        json.dump(sentences2location2valuesDiscarded, out,indent=4)
