'''

Charts the effect of thresholding across the development set for both APE thresholds and cost thresholds.

data/output/zero/arow_test/test2/summaryEvaluation.csv
figures/
data/output/zero/arow_test/results2/summaryEvaluationCost.csv

'''

from operator import itemgetter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
import random
from random import randint
import matplotlib.ticker as mtick
from mpl_toolkits.mplot3d import axes3d
import sys
from textwrap import wrap
import os

random.seed(11)

def costThresholdSlopeCharter3D(inputfile,outfile):

    resultCSV = pd.read_csv(inputfile).sort_values(by='ape_threshold')

    resultCSVAPE = resultCSV[(resultCSV.score_threshold=='no_prob_threshold')][['f1','model','cost_threshold','sigmoid_slope']].sort_values(by='model')

    openy = resultCSVAPE[resultCSVAPE.model.str.contains('costThreshold_open_cost_2')].copy()
    closedy = resultCSVAPE[resultCSVAPE.model.str.contains('costThreshold_closed_cost_2')].copy()


    openy.loc[:, 'model'] = 'open'
    closedy.loc[:, 'model'] = 'closed'

    # print openy

    open_f1 = openy.pivot_table('f1', 'cost_threshold', 'sigmoid_slope', fill_value=0).as_matrix().flatten()

    open_sigmoid = openy.groupby("cost_threshold").sigmoid_slope.apply(pd.Series.reset_index, drop=True).unstack().values.flatten()

    open_cost_threshold = openy.groupby("sigmoid_slope").cost_threshold.apply(pd.Series.reset_index, drop=True).unstack().values.flatten()

    closed_f1 = closedy.pivot_table('f1', 'cost_threshold', 'sigmoid_slope', fill_value=0).as_matrix().flatten()

    closed_sigmoid = closedy.groupby("cost_threshold").sigmoid_slope.apply(pd.Series.reset_index, drop=True).unstack().values.flatten()

    closed_cost_threshold = closedy.groupby("sigmoid_slope").cost_threshold.apply(pd.Series.reset_index, drop=True).unstack().values.flatten()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(open_sigmoid, open_cost_threshold, open_f1, c='r', marker='o',label='Open Evaluation: Cost-Sensitive Model')
    ax.scatter(closed_sigmoid, closed_cost_threshold, closed_f1, c='b', marker='^',label='Closed Evaluation: Cost-Sensitive Model')

    ax.set_xlabel('Sigmoid')
    ax.set_ylabel('Cost Threshold')
    ax.set_zlabel('F1')

    openlabel = "\n".join(wrap('Open Evaluation: Cost-Sensitive Model'))
    closedlabel = "\n".join(wrap('Closed Evaluation: Cost-Sensitive Model'))
    title = "\n".join(wrap('Dev Set: Interaction of Sigmoid Slope & Cost Threshold'))

    plt.title(title,fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.tick_params(axis='both', which='minor', labelsize=6)
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:1.0f}%'.format(x*100) for x in vals])
    ax.set_zticklabels(['{:1.0f}%'.format(x*100) for x in vals])
    ax.set_zticklabels(ax.get_zticks(),fontsize=6)

    # fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
    # xticks = mtick.FormatStrFormatter(fmt)
    # ax.xaxis.set_major_formatter(xticks)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    L = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              fancybox=True, shadow=True, ncol=2,fontsize=8)

    plt.savefig(outfile)

def costThresholdCharter(inputfile, outfile):

    resultCSV = pd.read_csv(inputfile).sort_values(by='cost_threshold')

    resultCSVAPE = resultCSV[(resultCSV.score_threshold=='no_prob_threshold') & (resultCSV.sigmoid_slope==1)][['f1','model','cost_threshold']].sort_values(by='cost_threshold')

    openy = resultCSVAPE[resultCSVAPE.model.str.contains('costThreshold_open_cost_2')].copy()
    closedy = resultCSVAPE[resultCSVAPE.model.str.contains('costThreshold_closed_cost_2')].copy()

    openy.loc[:, 'model'] = 'open'
    closedy.loc[:, 'model'] = 'closed'


    x = openy[['cost_threshold']]
    y1 = openy[['f1']]
    y2 = closedy[['f1']]

    plt.figure()

    fig,ax = plt.subplots()

    openlabel = "\n".join(wrap('Open Evaluation: Cost-Sensitive Model'))
    closedlabel = "\n".join(wrap('Closed Evaluation: Cost-Sensitive Model'))
    title = "\n".join(wrap('Dev Set: Cost Thresholding Effect: Sigmoid Slope 1'))

    plt.plot(x,y1,label=openlabel)
    plt.plot(x,y2,label=closedlabel)
    plt.xlabel('Cost Thresholds')
    plt.ylabel('F1',rotation=0,labelpad=15)
    plt.title(title,fontsize = 10)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:1.0f}%'.format(x*100) for x in vals])
    # fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
    # xticks = mtick.FormatStrFormatter(fmt)
    # ax.xaxis.set_major_formatter(xticks)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    L = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              fancybox=True, shadow=True, ncol=2,fontsize=8)

    # plt.tight_layout()

    plt.savefig(outfile)

def slopeThresholdCharter(inputfile, outfile):

    resultCSV = pd.read_csv(inputfile).sort_values(by='sigmoid_slope')

    resultCSVAPE = resultCSV[(resultCSV.score_threshold=='no_prob_threshold') & (resultCSV.cost_threshold==0.001)][['f1','model','sigmoid_slope']].sort_values(by='sigmoid_slope')

    openy = resultCSVAPE[resultCSVAPE.model.str.contains('costThreshold_open_cost_2')].copy()
    closedy = resultCSVAPE[resultCSVAPE.model.str.contains('costThreshold_closed_cost_2')].copy()

    openy.loc[:, 'model'] = 'open'
    closedy.loc[:, 'model'] = 'closed'


    x = openy[['sigmoid_slope']]
    y1 = openy[['f1']]
    y2 = closedy[['f1']]

    plt.figure()

    fig,ax = plt.subplots()

    openlabel = "\n".join(wrap('Open Evaluation: Cost-Sensitive Model'))
    closedlabel = "\n".join(wrap('Closed Evaluation: Cost-Sensitive Model'))
    title = "\n".join(wrap('Dev Set: Sigmoid Slope Effect: Cost Threshold 0.1%'))

    plt.plot(x,y1,label=openlabel)
    plt.plot(x,y2,label=closedlabel)
    plt.xlabel('Sigmoid Slopes')
    plt.ylabel('F1',rotation=0,labelpad=15)
    plt.title(title,fontsize = 10)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:1.0f}%'.format(x*100) for x in vals])
    # fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
    # xticks = mtick.FormatStrFormatter(fmt)
    # ax.xaxis.set_major_formatter(xticks)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    L = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              fancybox=True, shadow=True, ncol=2,fontsize=8)

    # plt.tight_layout()

    plt.savefig(outfile)

def apeThresholdCharter(inputfile, outfile):

    resultCSV = pd.read_csv(inputfile).sort_values(by='ape_threshold')

    resultCSVAPE = resultCSV[(resultCSV.ape_threshold!='no_mape_threshold') & ((resultCSV.model=="open_multi_logit_threshold_wordgrams") | (resultCSV.model=="closed_multi_logit_threshold_wordgrams")) & (resultCSV.probThreshold=='no_probability_threshold')].sort_values(by='model')


    resultCSVF1 = resultCSVAPE[['f1','precision','recall','accuracy','model','ape_threshold']].sort_values(by='ape_threshold')

    # print resultCSVF1

    openy = resultCSVF1[(resultCSVF1.model == 'open_multi_logit_threshold_wordgrams')]

    closedy = resultCSVF1[(resultCSVF1.model == 'closed_multi_logit_threshold_wordgrams')]

    print openy

    print closedy

    x = openy[['ape_threshold']]
    y1 = openy[['f1']]
    y2 = closedy[['f1']]

    plt.figure()

    fig,ax = plt.subplots()

    openlabel = "\n".join(wrap('Open Evaluation: Distantly Supervised Model'))
    closedlabel = "\n".join(wrap('Closed Evaluation: Distantly Supervised Model'))
    title = "\n".join(wrap('Dev Set: APE Thresholding Effect: Words + Dependency Bigram Models'))

    plt.plot(x,y1,label=openlabel)
    plt.plot(x,y2,label=closedlabel)
    plt.xlabel('APE Thresholds')
    plt.ylabel('F1',rotation=0,labelpad=15)
    plt.title(title,fontsize = 10)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:1.0f}%'.format(x*100) for x in vals])
    # fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
    # xticks = mtick.FormatStrFormatter(fmt)
    # ax.xaxis.set_major_formatter(xticks)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    L = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              fancybox=True, shadow=True, ncol=2,fontsize=8)

    # plt.tight_layout()

    plt.savefig(outfile)


if __name__ == "__main__":

    directory = sys.argv[2]

    # apeThresholdCharter(sys.argv[1],os.path.join(directory,'apeThresholds.png'))

    # costThresholdSlopeCharter3D(sys.argv[3],os.path.join(directory,'costThresholds.png'))

    costThresholdCharter(sys.argv[3],os.path.join(directory,'costThresholdsOnly.png'))

    slopeThresholdCharter(sys.argv[3],os.path.join(directory,'sigmoidSlopes.png'))
