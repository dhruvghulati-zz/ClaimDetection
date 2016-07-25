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
TODO - this needs to really sort out the problems with too many training instances for region/value slots e.g. duplicates for Democratic Republic of Congo, however have fixed problem with parsed sentence including duplicate slots (so better for bag of words)
http://stackoverflow.com/questions/38509239/need-to-remove-items-from-both-a-list-and-a-dictionary-of-tuple-value-pairs-at-s
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


def fixSlots(sampleTokens,tokenIDs2location,tokenIDs2number):
    # print "Old tokens are",sampleTokens
    #
    # print "Old Location token IDs are: ", tokenIDs2location
    # print "Old Number token IDs are: ", tokenIDs2number
    # print "Location is: ", location
    # print "Number is: ", number

    newTokens = []
    newnumberTokenIDs = {}
    newlocationTokenIDs = {}

    new_ind = 0
    skip = False

    for ind in range(len(sampleTokens)):
        if skip:
            skip=False
            continue

        for loc_ind in tokenIDs2location.keys():
            if ind in loc_ind:
                newTokens.append(sampleTokens[ind+1])
                newlocationTokenIDs[(new_ind,)] = tokenIDs2location[loc_ind]
                new_ind += 1
                if len(loc_ind) > 1: # Skip next position if there are 2 elements in a tuple
                    skip = True
                break
        else:
            for num_ind in tokenIDs2number.keys():
                if ind in num_ind:
                    newTokens.append(sampleTokens[ind])
                    newnumberTokenIDs[(new_ind,)] = tokenIDs2number[num_ind]
                    new_ind += 1
                    if len(num_ind) > 1:
                        skip = True
                    break
            else:
                newTokens.append(sampleTokens[ind])
                new_ind += 1

    # print "New Location token IDs are: ", newlocationTokenIDs
    # print "New Number token IDs are: ", newnumberTokenIDs
    # # print "Location is: ", location
    # # print "Number is: ", number
    # print "New Tokens are", newTokens

    return newTokens, newlocationTokenIDs,newnumberTokenIDs


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
    for jsonFileName in itertools.islice(jsonFiles , 0, len(jsonFiles)):
    # for fileCounter, jsonFileName in enumerate(jsonFiles):
        # For each file in the HTML JSON
        print "processing " + jsonFileName
        with codecs.open(jsonFileName) as jsonFile:
            parsedSentences = json.loads(jsonFile.read())
        for sentence in parsedSentences:

            # print sentence

            #
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

                # sampleTokens = sample.split()

                # newTokens, newTokenIDs2location,newTokenIDs2number = fixSlots(sampleTokens,tokenIDs2location,tokenIDs2number)

                # for locationTokenIDs, location in tokenIDs2location.items():
                #     for numberTokenIDs, number in tokenIDs2number.items():
                #         print "Location token IDs are: ", locationTokenIDs
                #         print "Number token IDs are: ", numberTokenIDs
                #         print "Location is: ", location
                #         print "Number is: ", number

                # TODO - run a function to create new tokens

                for locationTokenIDs, location in tokenIDs2location.items():
                    for numberTokenIDs, number in tokenIDs2number.items():



                        sentenceDict = {}

                        if len(numberTokenIDs)>3 or len(locationTokenIDs)>3:
                            sentenceDict["dense"]=True
                        else:
                            sentenceDict["dense"]=False

                        sampleTokens = sample.split()

                        # newTokens, newLocationTokenIDs,newNumberTokenIDs = fixSlots(sampleTokens,tokenIDs2location,tokenIDs2number)

                        sentenceDict["sentence"] = sample

                        sentenceDict["location-value-pair"] = {location:number}

                        prevNoId = 0
                        prevLocId = 0

                        for locationTokenID in locationTokenIDs:
                            for numberTokenID in numberTokenIDs:


                                # print locationTokenID

                                # TODO - recalculate the tokens


                                sampleTokens[numberTokenID] = "NUMBER_SLOT"
                                sampleTokens[locationTokenID] = "LOCATION_SLOT"


                                # print "New tokens",newTokens

                                slotSentence = (" ").join(sampleTokens)
                                sentenceDict["parsedSentence"] = slotSentence

                        sentences2location2values["sentences"].append(sentenceDict)
                        sentences2location2valuesSlots.append(sentenceDict)

    # Filtering the sentenceSlots afterwards

    # TODO - need to also delete training instances with multiple region value-pairs, and do this for all countries
    for i,(sentence,finalSentence) in enumerate(zip(sentences2location2valuesSlots,sentences2location2values["sentences"])):
        sampleTokens = sentence['parsedSentence'].split()
        # print "Sentence is ",sentence['parsedSentence']
        # print "Final sentence is",finalSentence
        # finalSampleTokens = finalSentence['parsedSentence'].split()
        # print "Old sample tokens are",sampleTokens,"\n"
        newTokens = []
        for i,token in enumerate(sampleTokens):
            if i>0 and ((token == "LOCATION_SLOT" and sampleTokens[i-1]=="LOCATION_SLOT") or (token == "NUMBER_SLOT" and sampleTokens[i-1]=="NUMBER_SLOT")):
                continue
            else:
                newTokens.append(token)

        sentence['parsedSentence']=(' ').join(newTokens)
        finalSentence['parsedSentence']=(' ').join(newTokens)

        # print "New sentence",sentence['parsedSentence'],"\n"

        locationCount=0
        numberCount=0

        # Remove items with too many location and number slots
        denseSentence = False
        tooManySlots = sentence["dense"]
        analyser = Text(newTokens)

        for token in newTokens:
            # if token =="LOCATION_SLOT":
            #     locationCount+=1
            # elif token =="NUMBER_SLOT":
            #     numberCount+=1
            # if locationCount>3 or numberCount>3:
            #     # print "Too many tokens are",sampleTokens
            #     # print "Sentence is",sample
            # # print "Number of locations", len(tokenIDs2location)
            # # print "Number of values", len(tokenIDs2number)
            #     tooManySlots = True
            #     break
            wordDensity = float(analyser.count(token))/float(len(newTokens))
            # print wordDensity
            if wordDensity>wordDensityThreshold:
                # print "wordDensity is",wordDensity
                # print "threshold is",wordDensityThreshold
                # print "Too dense tokens are",sampleTokens
                # print "Sentence is",sample
                denseSentence = True
                # "Exiting loop..."
                break

        if tooManySlots or denseSentence:
            # print "Dense sentence is",sentence["sentence"]
            sentences2location2valuesDiscarded.append(sentence)
            del sentences2location2valuesSlots[i]


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
