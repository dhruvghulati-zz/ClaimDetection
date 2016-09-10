'''
# TODO - goes through the files in the predict and probabilistic versions and creates a) the scores for the predict versions and b) loops through given a cycle all the different combinations of bins of probability thresholds to use, saving the threshold in the name of the file and appending to the sequence. Need to get access to the name of the file


data/output/zero/arow_test/open_label_mapping.txt
data/output/zero/arow_test/closed_label_mapping.txt
data/output/zero/arow_test/open_label_mapping_threshold.txt
data/output/zero/arow_test/closed_label_mapping_threshold.txt
data/output/devLabels.json
data/output/zero/arow_test/predict/
data/output/zero/arow_test/probPredict/
data/output/zero/arow_test/results/

'''

from itertools import chain
import numpy as np
import sys
import os
import json
import pandas as pd
from sklearn.metrics import precision_score, precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import collections

def labelConvert(modelMatrix):
    final_predictions = {}
    for model,predictions in modelMatrix.items():
        if model.startswith("threshold") or model.startswith("thresholdCostThreshold"):
            if str(model).split('_')[1] == "open":
                final_predictions[model]=[openInvMappingThreshold[str(i)] for i in predictions]
            else:
                final_predictions[model]=[closedInvMappingThreshold[str(i)] for i in predictions]
        else:
            if str(model).split('_')[1]=="open" or str(model).split('_')[0] == "open":
                final_predictions[model]=[openInvMapping[str(i)] for i in predictions]
            else:
                final_predictions[model]=[closedInvMapping[str(i)] for i in predictions]
    return final_predictions


def probLabelConvert(modelMatrix):
    final_predictions = {}
    for model,predictionDict in modelMatrix.items():
        final_predictions[model]={}
        for threshold,predictions in predictionDict.items():
            if isinstance(threshold,float):
                final_predictions[model][threshold]=[]
                if model.startswith("threshold") or model.startswith("thresholdCostThreshold"):
                    if str(model).split('_')[1] == "open":
                        final_predictions[model][threshold]=[openInvMappingThreshold[str(i)] if i in openInvMappingThreshold.keys() else i for i in predictions]
                    else:
                        final_predictions[model][threshold]=[closedInvMappingThreshold[str(i)] if i in closedInvMappingThreshold.keys() else i for i in predictions]
                else:
                    if str(model).split('_')[1]=="open" or str(model).split('_')[0] == "open":
                        final_predictions[model][threshold]=[openInvMapping[str(i)] if i in openInvMapping.keys() else i for i in predictions ]
                    else:
                        final_predictions[model][threshold]=[closedInvMapping[str(i)] if i in closedInvMapping.keys() else i for i in predictions]
    return final_predictions




def convert(data):
    if isinstance(data, basestring):
        return str(data)
    elif isinstance(data, collections.Mapping):
        return dict(map(convert, data.iteritems()))
    elif isinstance(data, collections.Iterable):
        return type(data)(map(convert, data))
    else:
        return data

'''
Load in the relevant data
'''
with open(sys.argv[1]) as mapping_file:
    openInvMapping = json.loads(mapping_file.read())

openInvMapping = convert(openInvMapping)

with open(sys.argv[2]) as closed_map_file:
    closedInvMapping = json.loads(closed_map_file.read())

closedInvMapping = convert(closedInvMapping)

with open(sys.argv[3]) as mapping_file_threshold:
    openInvMappingThreshold = json.loads(mapping_file_threshold.read())

openInvMappingThreshold = convert(openInvMappingThreshold)

with open(sys.argv[4]) as closed_map_file_threshold:
    closedInvMappingThreshold = json.loads(closed_map_file_threshold.read())

closedInvMappingThreshold = convert(closedInvMappingThreshold)

with open(sys.argv[5]) as finalTestSentences:
    finalTestSentences = json.loads(finalTestSentences.read())

# Load in the non probabilistic predictions
cost_predictions = {}
for (dirpath, dirnames, filenames) in os.walk(sys.argv[6]):
    for filename in filenames:
        if filename.endswith('.predict'):
            model = filename.split(".predict")[0]
            # print "Model is",model
            # print "Directory is",dirpath
            # print "Filename is", filename
            cost_predictions[model] = map(int,[i[0] for i in np.loadtxt(os.sep.join([dirpath, filename]))])

cost_predictions = labelConvert(cost_predictions)

# print cost_predictions

# Load in the probabilistic predictions
prob_predictions = {}
for (dirpath, dirnames, filenames) in os.walk(sys.argv[7]):
    for filename in filenames:
        if filename.endswith('.probpredict'):
            model = filename.split(".probpredict")[0]
            prob_predictions[model] = {'prediction':[]}
            with open(os.sep.join([dirpath, filename])) as f:
                for line in f:
                    dict = {}
                    line = line.split()[:][:-1]
                    for kv in line:
                        dict[kv.split(":")[0]]=float(kv.split(":")[1])
                    prob_predictions[model]['prediction'].append(dict)

# Now calculate the correct max and min threshold for each item

# 0.0001 0.001 0.0050 0.01 0.02 0.04 0.05 0.06 0.075 0.1 0.15 0.30 0.40 0.50 0.75 1.0

for model,data in prob_predictions.items():
    flattened = list(chain(*(d.values() for d in data['prediction'])))
    # print len(data['prediction'])
    maxScore = max(flattened)
    minScore = min(flattened)
    data['normal_prediction'] = []
    for i,dict in enumerate(data['prediction']):
        temp_dict = {key:((value - minScore)/(maxScore - minScore)) for key,value in dict.iteritems()}
        data['normal_prediction'].append(temp_dict)
    # print len(data['normal_prediction'])

    # data['score_range']=[]
    # Calculate the min and max
    # flattened = list(chain(*(d.values() for d in data['prediction'])))

    # print "Scores are",flattened
    # flattenedPct = [(i-min(flattened))/(max(flattened)-min(flattened))for i in flattened]
    # openThresholdProb = (1/float(len(openInvMappingThreshold.keys())))
    # closedThresholdProb = (1/float(len(closedInvMappingThreshold.keys())))
    # openProb = (1/float(len(openInvMapping.keys())))
    # closedProb = (1/float(len(closedInvMapping.keys())))
    # probThresholds = sorted([openThresholdProb,closedThresholdProb,openProb,closedProb,0.0001,0.001,0.005,0.01,0.02,0.04,0.05,0.06,0.075,0.10,0.15,0.30,0.40,0.50,0.75,1])
    # # print "Prob thresholds are", probThresholds
    # data['score_range'] = probThresholds
    # flattenedPct = np.percentile(flattened,probThresholds)
    # data['score_range'] = flattenedPct
    # print "Flattened scores are",flattenedPct

# probThresholds = [10,15,20,25,30,35,40,50,75,100,200,300,250,500,750,1000]

openThresholdProb = (1/float(len(openInvMappingThreshold.keys())))
closedThresholdProb = (1/float(len(closedInvMappingThreshold.keys())))
openProb = (1/float(len(openInvMapping.keys())))
closedProb = (1/float(len(closedInvMapping.keys())))
probThresholds = sorted([openThresholdProb,closedThresholdProb,openProb,closedProb,0.0001,0.001,0.005,0.01,0.02,0.04,0.05,0.06,0.075,0.10,0.15,0.30,0.40,0.50,0.75,1.0])


for score in probThresholds:
    for model,data in prob_predictions.items():
        # print "Score is",score
        # # print "Ranges are",data['score_range']
        # for score in data['score_range']:
        #     for model,data in prob_predictions.items():
        data[score] = []
        for inner_dict in data['normal_prediction']:
            # print inner_dict
            # print "Score is",score
            label_prediction = max(inner_dict, key=inner_dict.get)
            score_prediction = max(inner_dict.itervalues())
            # print "Score prediction is",score_prediction
            # print "Prob threshold is", float(i)
            if score_prediction>float(score):
                data[score].append(label_prediction)
            else:
                data[score].append("below_score")
        # print "Score is",len(data[score])

# for model,data in prob_predictions.items():
#     for i in data['score_range']:
#         print data[i]

# Now we convert the labels
prob_predictions = probLabelConvert(prob_predictions)

# '''
# Test if the probability label is having an effect
# '''
#
# with open(os.path.join(sys.argv[8],'prob_models.json'),"wb") as f:
#     json.dump(prob_predictions,f,indent=4)

# Load in the test data
test = pd.DataFrame(finalTestSentences)

# These are the ground truths
y_multi_true = np.array(test['property'])
y_true_claim = np.array(test['claim'])

binary_cslr_predictions = {}

categorical_data = {}

for model,predictions in cost_predictions.iteritems():
    if predictions:
        categorical_data[(model,'precision')]=precision_recall_fscore_support(y_multi_true,predictions,labels=list(set(y_multi_true)),average=None)[0]
        categorical_data[(model,'recall')]=precision_recall_fscore_support(y_multi_true,predictions,labels=list(set(y_multi_true)),average=None)[1]
        categorical_data[(model,'f1')]=precision_recall_fscore_support(y_multi_true,predictions,labels=list(set(y_multi_true)),average=None)[2]
        binary_cslr_predictions[model]=[]
        for predict,true in zip(predictions,y_multi_true):
            if predict==true:
                binary_cslr_predictions[model].append(1)
            else:
                binary_cslr_predictions[model].append(0)


binary_prob_predictions = {}

for model,predDict in prob_predictions.iteritems():
    binary_prob_predictions[model]={}
    for threshold, predictions in predDict.iteritems():
        binary_prob_predictions[model][threshold]=[]
        if predictions:
            categorical_data[(model+"_"+str(threshold),'precision')]=precision_recall_fscore_support(y_multi_true,predictions,labels=list(set(y_multi_true)),average=None)[0]
            categorical_data[(model+"_"+str(threshold),'recall')]=precision_recall_fscore_support(y_multi_true,predictions,labels=list(set(y_multi_true)),average=None)[1]
            categorical_data[(model+"_"+str(threshold),'f1')]=precision_recall_fscore_support(y_multi_true,predictions,labels=list(set(y_multi_true)),average=None)[2]
            for predict,true in zip(predictions,y_multi_true):
                if predict==true:
                    binary_prob_predictions[model][threshold].append(1)
                else:
                    binary_prob_predictions[model][threshold].append(0)

# model_path = os.path.join(sys.argv[8] + '/models.txt')
#
# with open(model_path, "wb") as out:
#     json.dump(binary_prob_predictions, out)


categorical_data = pd.DataFrame(categorical_data,index=[item.split('/')[3] for item in list(set(y_multi_true))])

categoricalPath = os.path.join(sys.argv[8] + 'categoricalResults.csv')

categorical_data.to_csv(path_or_buf=categoricalPath,encoding='utf-8')

# TODO This was an issue on command line - change to [2] if on command line and 8 if not
testSet = str(os.path.splitext(sys.argv[5])[0]).split("/")[2]

# Now we write our precision F1 etc to an Excel file
summaryDF = pd.DataFrame()


def evaluation(trueLabels, evalLabels, test_set, apeThreshold, costThreshold,probThreshold, inputBias, inputSlope):
    global summaryDF

    precision = precision_score(trueLabels, evalLabels)
    recall = recall_score(trueLabels, evalLabels)
    f1 = f1_score(trueLabels, evalLabels)
    accuracy = accuracy_score(trueLabels, evalLabels)

    data = {'precision': [precision],
            'recall': [recall],
            'f1': [f1],
            'accuracy': [accuracy],
            'evaluation set': [test_set],
            'ape_threshold': [apeThreshold],
            'cost_threshold': [costThreshold],
            'score_threshold': [probThreshold],
            'sigmoid_bias': [inputBias],
            'sigmoid_slope': [inputSlope]
            }

    DF = pd.DataFrame(data)

    summaryDF = pd.concat([summaryDF, DF])

for model,result in binary_cslr_predictions.iteritems():
    ape_threshold = model.split("_")[::-1][4]
    cost_threshold = model.split("_")[::-1][3]
    bias = model.split("_")[::-1][2]
    slope = model.split("_")[::-1][1]
    # print ape_threshold
    # print cost_threshold
    evaluation(y_true_claim, result, testSet, ape_threshold,cost_threshold,"no_prob_threshold", bias, slope)

for model,resultDict in binary_prob_predictions.iteritems():
    ape_threshold = model.split("_")[::-1][4]
    cost_threshold = model.split("_")[::-1][3]
    bias = model.split("_")[::-1][2]
    slope = model.split("_")[::-1][1]
    for prob,result in resultDict.items():
        evaluation(y_true_claim, result, testSet, ape_threshold,cost_threshold,prob, bias, slope)

# Now generate keys for prob predictions
probList = []
for model,dict in binary_prob_predictions.items():
    for threshold,inner_dict in dict.items():
        probList.append(model + "_" + str(threshold))

columns = list(binary_cslr_predictions.keys()) + probList

summaryDF.index = columns

F1Path = os.path.join(sys.argv[8] + 'summaryEvaluationCost.csv')

try:
    if os.stat(F1Path).st_size > 0:
        with open(F1Path, 'a') as f:
            summaryDF.to_csv(path_or_buf=f, encoding='utf-8', mode='a', header=False)
            f.close()
    else:
        print "empty file"
        with open(F1Path, 'w+') as f:
            summaryDF.to_csv(path_or_buf=f, encoding='utf-8')
            f.close()
except OSError:
    print "No file"
    with open(F1Path, 'w+') as f:
        summaryDF.to_csv(path_or_buf=f, encoding='utf-8')
        f.close()