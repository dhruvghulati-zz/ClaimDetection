'''
# so this script takes as input a dictionary in json with the following structure:
# dep or string pattern : {location1:[values], location2:[values]}, etc.
# and does the following kinds of filtering:
# - removes locations that have less than one value for a pattern
# - removes patterns for which location lists are all over the place (high stdev)
# - removes patterns that have fewer than arg1 location

# The second argument is a list of (FreeBase) region names to their aliases which will
# to bring condense the matrix (UK and U.K. becoming the same location), but also they
# prepare us for experiments 
'''

import json
import numpy as np
import sys
from collections import OrderedDict
import yaml
import itertools
import operator
from operator import itemgetter
import pprint

'''
TODO - this needs to account for any further cleaning beyond aliasing we need to do, for example not including anything where the value is 0.
'''

# data/output/sentenceRegionValue.json
# data/aliases.json
# data/sentenceMatrixFiltered.json
# data/output/sentenceSlotsFiltered.json
# data/output/uniqueSentenceLabels.json

# We distinguish between the two by re- quiring each region-pattern combination to have appeared at least twice.

# python src/main/sentenceMatrixFiltering.py data/output/sentenceRegionValue.json data/aliases.json data/sentenceMatrixFiltered.json


# helps detect errors
np.seterr(all='raise')

# load the file
with open(sys.argv[1]) as jsonFile:
    pattern2locations2values = json.loads(jsonFile.read())
    # pattern2locations2values = yaml.safe_load(jsonFile)

# load the file
with open(sys.argv[4]) as sentenceSlotsFull:
    fullSentenceSlots = json.loads(sentenceSlotsFull.read())
    # fullSentenceSlots = yaml.safe_load(sentenceSlotsFull)

print "Model sentences before filtering:", len(pattern2locations2values['sentences'])
print "labelling sentences before filtering:", len(fullSentenceSlots)

getvals = operator.itemgetter(u"parsedSentence", u"sentence",u"location-value-pair")
getvalsLabels = operator.itemgetter(u"sentence")

pattern2locations2values['sentences'].sort(key=getvals)
fullSentenceSlots.sort(key=getvalsLabels)

result = []
for k, g in itertools.groupby(pattern2locations2values['sentences'], getvals):
    result.append(g.next())
pattern2locations2values['sentences'][:] = result

result = []
for k, g in itertools.groupby(fullSentenceSlots, getvalsLabels):
    result.append(g.next())
fullSentenceSlots[:] = result

print "Unique sentences after filtering:", len(pattern2locations2values['sentences'])
print "Unique labelling sentences after filtering:", len(fullSentenceSlots)

# Now remove any sentences with values of 0:


finalpattern2locations2values={}
finalpattern2locations2values['sentences']=[]

for i,sentence in enumerate(pattern2locations2values['sentences']):
    for key, value in sentence['location-value-pair'].iteritems():
        if value!=0.0:
            finalpattern2locations2values['sentences'].append(sentence)


print "Unique sentences after deleting 0 values:", len(finalpattern2locations2values['sentences'])

#load the file
print "Loading the aliases file\n"
with open(sys.argv[2]) as jsonFile:
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

# ok, let's traverse now all the patterns and any locations we find we match them case independently to the aliases and replace them with the location

print "Applying aliases for model\n"

for index, dataTriples in enumerate(finalpattern2locations2values["sentences"]):
    # print index
    # so here are the locations
    # we must be careful in case two or more locations are collapsed to the same region
    for location, value in dataTriples['location-value-pair'].items():
        # print location
        region = location
    #         # if the location has an alias
        if location in alias2region:
            # get it
            region = alias2region[location]
            # print "New region is ", region
        elif location.lower() in alias2region:
            region = alias2region[location.lower()]
            # print "New region is ", region
        # if we haven't added it to the regions
        dataTriples['location-value-pair']= {region:value}
        # regions2values.append({region: value})
# print pattern2locations2values

# print "Applying aliases for manual labelling\n"
#
# for index, dataTriples in enumerate(fullSentenceSlots):
#     # print index
#     # so here are the locations
#     # we must be careful in case two or more locations are collapsed to the same region
#
#     for i,location in enumerate(dataTriples['regions']):
#         # print location
#         # print value
#         region = location
#     #         # if the location has an alias
#         if location in alias2region:
#             # get it
#             region = alias2region[location]
#             # print "New region is ", region
#         elif location.lower() in alias2region:
#             region = alias2region[location.lower()]
#             # print "New region is ", region
#         # if we haven't added it to the regions
#         dataTriples['regions'][i] = region
#         # location = region
#         # np.put(dataTriples['regions'], i, region)
#         # regions2values.append({region: value})
# # print pattern2locations2values



print "Writing to filtered file for model\n"

with open(sys.argv[3], "wb") as out:
    json.dump(finalpattern2locations2values, out,indent=4)

print "Writing to filtered file for manual labelling\n"

with open(sys.argv[5], "wb") as out:
    json.dump(fullSentenceSlots, out,indent=4)

