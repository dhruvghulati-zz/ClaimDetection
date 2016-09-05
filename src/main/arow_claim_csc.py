'''

This outputs prediction files with and without probabilities for separate folders using the AROW algorithm, preserving the name of the file.

Needs to be run on a bash script.

python src/main/arow_claim_csc.py data/output/zero/arow_test/closed_cost_1.dat data/output/zero/arow_test/test.dat data/output/zero/arow_test/predict/closed_cost_1.predict data/output/zero/arow_test/probPredict/closed_cost_1.probpredict

'''

import arow
import sys
import numpy as np

# TODO - make this have parameters that can be adjusted
def sigmoid(v):
    return ((1 / (1 + np.exp(-v)))*2)-1

def min_max(lst):
    values = [v for v in lst if v is not None]
    return min(values), max(values)

def normalize(v, least, most):
    return 1.0 if least == most else float(v - least) / (most - least)

def normalize_dicts_local(lst):
    spans = [min_max(dic.values()) for dic in lst]
    return [{key: normalize(val,*span) for key,val in dic.iteritems()} for dic,span in zip(lst,spans)]

def normalize_dicts_local_sigmoid(lst):
    return [{key: sigmoid(val) for key,val in dic.iteritems()} for dic in lst]

if __name__ == "__main__":

    # Checks on command line if it is actually containing data
    trainDataLines = open(sys.argv[1]).readlines()
    # test.dat
    testDataLines = open(sys.argv[2]).readlines()
    # open_cost_1.predict
    predictFile = sys.argv[3]

    probPredictFile = sys.argv[4]

    train_data = [arow.train_instance_from_svm_input(line) for line in trainDataLines]
    test_data = [arow.test_instance_from_svm_input(line) for line in testDataLines]
    cl = arow.AROW()
    # print [cl.predict(d).label for d in test_data]
    # print [d.costs for d in test_data]

    cl.train(train_data)
    # cl.probGeneration()
    # ,probabilities=False
    predictions = [cl.predict(d, verbose=True).label for d in test_data]
    predictionScores = [cl.predict(d, verbose=True).label2score for d in test_data]

    # Now also normalise the scores

    # print predictions
    # for i,prediction in enumerate(predictions):
    #     print prediction
    # print [cl.predict(d, verbose=True).featureValueWeights for d in test_data]
    # print [d.costs for d in test_data]

    f = open(predictFile, 'w')

    for i, prediction in enumerate(predictions):
        line = str(int(prediction)) + " " + str(i)
        f.write(line+"\n")

    f.close()

    fp = open(probPredictFile, 'w')

    # Now normalise the scores
    # predictionScores = normalize_dicts_local(predictionScores)

    for i, prediction in enumerate(predictionScores):
        line = ""
        for label, score in prediction.iteritems():
            line += str(int(label)) + ":" + str(score) + " "
        fp.write(line+ str(i)+"\n")

    fp.close()