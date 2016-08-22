'''

python src/main/arow_claim_csc.py data/output/zero/arow_test/closed_cost_1.dat data/output/zero/arow_test/test.dat data/output/zero/arow_test/closed_cost_1.predict

'''

import arow
import sys

if __name__ == "__main__":

    trainDataLines = open(sys.argv[1]).readlines()
    # test.dat
    testDataLines = open(sys.argv[2]).readlines()
    # open_cost_1.predict
    predictFile = sys.argv[3]

    train_data = [arow.train_instance_from_svm_input(line) for line in trainDataLines]
    test_data = [arow.test_instance_from_svm_input(line) for line in testDataLines]
    cl = arow.AROW()
    # print [cl.predict(d).label for d in test_data]
    # print [d.costs for d in test_data]

    cl.train(train_data)
    # cl.probGeneration()
    # ,probabilities=False
    predictions = [cl.predict(d, verbose=True).label2score for d in test_data]

    # print predictions
    for i,prediction in enumerate(predictions):
        print prediction
    # print [cl.predict(d, verbose=True).featureValueWeights for d in test_data]
    # print [d.costs for d in test_data]

    f = open(predictFile, 'w')

    for i, prediction in enumerate(predictions):
        line = str(int(prediction)) + " " + str(i)
        f.write(line+"\n")

    f.close()