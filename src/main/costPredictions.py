import numpy as np
import sys
import os
import json
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import collections

#
def convert(data):
    if isinstance(data, basestring):
        return str(data)
    elif isinstance(data, collections.Mapping):
        return dict(map(convert, data.iteritems()))
    elif isinstance(data, collections.Iterable):
        return type(data)(map(convert, data))
    else:
        return data

with open(sys.argv[1]) as model_file:
    cost_matrices = json.loads(model_file.read())

with open(sys.argv[2]) as mapping_file:
    openInvMapping = json.loads(mapping_file.read())

openInvMapping = convert(openInvMapping)

print openInvMapping

with open(sys.argv[3]) as closed_map_file:
    closedInvMapping = json.loads(closed_map_file.read())

closedInvMapping = convert(closedInvMapping)

with open(sys.argv[4]) as finalTestSentences:
    finalTestSentences = json.loads(finalTestSentences.read())

cost_predictions = {}

for model in cost_matrices.keys():
    print "Model is",model
    cost_predictions[model]=np.loadtxt(os.path.join(sys.argv[5]+model+".predict"))
    cost_predictions[model] = map(int,[i[0] for i in cost_predictions[model]])
    # cost_predictions[model] = [str(i) for i in cost_predictions[model]]
    print "Predictions are",len(cost_predictions[model])
    print "Predictions are",cost_predictions[model]
    print "Open mappings are",openInvMapping
    print "Closed mappings are",closedInvMapping

    if str(model).startswith("single"):
        if str(model).split('_')[1]=="open":
            cost_predictions[model]=[openInvMapping[str(i)] for i in cost_predictions[model]]
        else:
            cost_predictions[model]=[closedInvMapping[str(i)] for i in cost_predictions[model]]
    else:
        if str(model).split('_')[0]=="open":
            cost_predictions[model]=[openInvMapping[str(i)] for i in cost_predictions[model]]
        else:
            cost_predictions[model]=[closedInvMapping[str(i)] for i in cost_predictions[model]]


# print cost_predictions
#
# Load in the test data
test = pd.DataFrame(finalTestSentences)
threshold = test['threshold'][0]

# These are the ground truths

y_multi_true = np.array(test['property'])
y_true_claim = np.array(test['claim'])

binary_cslr_predictions = {}

for key,predictions in cost_predictions.iteritems():
    if predictions:
        binary_cslr_predictions[key]=[]
        for predict,true in zip(predictions,y_multi_true):
            # print "Prediction is",predict
            # print "True is ",true
            if predict==true:
                binary_cslr_predictions[key].append(1)
            else:
                binary_cslr_predictions[key].append(0)
        # print "Number of predictions are",len(binary_cslr_predictions[key])

print binary_cslr_predictions

# TODO This was an issue on command line - change to [2] if on command line and 8 if not
testSet = str(os.path.splitext(sys.argv[4])[0]).split("/")[2]

# # # TODO - need to create a per property chart
# Now we write our precision F1 etc to an Excel file
summaryDF = pd.DataFrame(columns=('precision', 'recall', 'f1', 'accuracy', 'evaluation set', 'threshold'))
# # 'probThreshold'
#
def evaluation(trueLabels, evalLabels, test_set, threshold):
    # ,probThreshold
    global summaryDF
    global trainingLabels
    global positiveOpenTrainingLabels
    global negativeOpenTrainingLabels
    global positiveClosedTrainingLabels
    global negativeClosedTrainingLabels
    global openTrainingClasses
    global openTrainingClassesThreshold
    global closedTrainingClasses
    global closedTrainingClassesThreshold

    precision = precision_score(trueLabels, evalLabels)
    recall = recall_score(trueLabels, evalLabels)
    f1 = f1_score(trueLabels, evalLabels)
    accuracy = accuracy_score(trueLabels, evalLabels)

    data = {'precision': [precision],
            'recall': [recall],
            'f1': [f1],
            'accuracy': [accuracy],
            'evaluation set': [test_set],
            'threshold': [threshold]
            # 'probThreshold': [probThreshold],
            # 'trainingLabels':[trainingLabels],
            # 'positiveOpenLabels':[positiveOpenTrainingLabels],
            # 'negativeOpenLabels':[negativeOpenTrainingLabels],
            # 'positiveClosedLabels':[positiveClosedTrainingLabels],
            # 'negativeClosedLabels':[negativeClosedTrainingLabels],
            # 'openTrainingClasses': [openTrainingClasses],
            # 'openTrainingClassesThreshold': [openTrainingClassesThreshold],
            # 'closedTrainingClasses':[closedTrainingClasses],
            # 'closedTrainingClassesThreshold': [closedTrainingClassesThreshold]
            }

    DF = pd.DataFrame(data)

    summaryDF = pd.concat([summaryDF, DF])

for model,result in binary_cslr_predictions.iteritems():
    # print "Model is ",model
    # print "True claim is",np.array(y_true_claim)
    # print "Result is",result

    evaluation(y_true_claim, result, testSet, threshold)

columns = list(binary_cslr_predictions.keys())

summaryDF.index = columns

print summaryDF
#
try:
    if os.stat(sys.argv[6]).st_size > 0:
        # df_csv = pd.read_csv(sys.argv[5],encoding='utf-8',engine='python')
        # summaryDF = pd.concat([df_csv,summaryDF],axis=1,ignore_index=True)
        with open(sys.argv[6], 'a') as f:
            # Need to empty file contents now
            summaryDF.to_csv(path_or_buf=f, encoding='utf-8', mode='a', header=False)
            f.close()
    else:
        print "empty file"
        with open(sys.argv[6], 'w+') as f:
            summaryDF.to_csv(path_or_buf=f, encoding='utf-8')
            f.close()
except OSError:
    print "No file"
    with open(sys.argv[6], 'w+') as f:
        summaryDF.to_csv(path_or_buf=f, encoding='utf-8')
        f.close()