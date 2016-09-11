'''
Calculate simple precision recall f1 for all predictions from textual patterns model by loading in filepath.

data/output/zero/andreasnp
'''
import csv
import os
import sys
import xlrd
from sklearn.metrics import precision_score, f1_score, recall_score
import pandas as pd
import numpy as np


def evaluation(filepath):

    y_pred = []
    y_true = []
        # TODO - Command line issue as I had hard coded the location of these files ../../, in command line remove
    for subdir, dirs, files in os.walk(filepath):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".tsv"):
                print "Filepath is",filepath
                df = pd.read_csv(filepath, sep='\t')
                for pred in np.array(df.prediction):
                    y_pred.append(pred)

                for true in np.array(df.claim):
                    y_true.append(true)

    return y_pred,y_true

if __name__ == "__main__":

    y_pred, y_true = evaluation(sys.argv[1])

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print "Precision: ",precision
    print "Recall: ",recall
    print "F1: ",f1