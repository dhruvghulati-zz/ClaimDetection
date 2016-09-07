from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import copy
import random
from random import randint

random.seed(11)

word_freq_1 = [('test', 510), ('Hey', 362), ("please", 753), ('take', 446), ('herbert', 325), ('live', 222), ('hate', 210), ('white', 191), ('simple', 175), ('harry', 172), ('woman', 170), ('basil', 153), ('things', 129), ('think', 126), ('bye', 124), ('thing', 120), ('love', 107), ('quite', 107), ('face', 107), ('eyes', 107), ('time', 106), ('himself', 105), ('want', 105), ('good', 105), ('really', 103), ('away',100), ('did', 100), ('people', 99), ('came', 97), ('say', 97), ('cried', 95), ('looked', 94), ('tell', 92), ('look', 91), ('world', 89), ('work', 89), ('project', 88), ('room', 88), ('going', 87), ('answered', 87), ('mr', 87), ('little', 87), ('yes', 84), ('silly', 82), ('thought', 82), ('shall', 81), ('circle', 80), ('hallward', 80), ('told', 77), ('feel', 76), ('great', 74), ('art', 74), ('dear',73), ('picture', 73), ('men', 72), ('long', 71), ('young', 70), ('lady', 69), ('let', 66), ('minute', 66), ('women', 66), ('soul', 65), ('door', 64), ('hand',63), ('went', 63), ('make', 63), ('night', 62), ('asked', 61), ('old', 61), ('passed', 60), ('afraid', 60), ('night', 59), ('looking', 58), ('wonderful', 58), ('gutenberg-tm', 56), ('beauty', 55), ('sir', 55), ('table', 55), ('turned', 54), ('lips', 54), ("one's", 54), ('better', 54), ('got', 54), ('vane', 54), ('right',53), ('left', 53), ('course', 52), ('hands', 52), ('portrait', 52), ('head', 51), ("can't", 49), ('true', 49), ('house', 49), ('believe', 49), ('black', 49), ('horrible', 48), ('oh', 48), ('knew', 47), ('curious', 47), ('myself', 47)]

word_freq_1 = sorted(word_freq_1, key=lambda x: x[1],reverse=True)

word_freq_2 = [((tuple[0], randint(1,500))) for i,tuple in enumerate(word_freq_1)]

word_freq_2 = sorted(word_freq_2, key=lambda x: x[1],reverse=True)

# word_freq_2.extend((('sheepish',400),('miscellaneous',98)))

# N = max(len(word_freq_2),len(word_freq_2))
N = 25
ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()

xticks_train = [i for i in xrange(0,N,2)]
xticks_test = [i for i in xrange(1,N,2)]
# print xticks_train
# print xticks_test

def autolabel(rects,labels):
    # attach some text labels
    for i,(rect,label) in enumerate(zip(rects,labels)):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                label,
                ha='left', va='bottom',fontsize=6,style='italic',rotation=45)

def tupleCounts2Percents(inputList):
    total = sum(x[1] for x in inputList)*1.0
    return [(x[0], 1.*x[1]/total) for x in inputList]


values_1_pct = map(itemgetter(1), tupleCounts2Percents(word_freq_1))[:25]
# values_1_pct = ['{:.1%}'.format(item)for item in values_1_pct]

values_2_pct = map(itemgetter(1), tupleCounts2Percents(word_freq_2))[:25]
# values_2_pct = ['{:.1%}'.format(item)for item in values_2_pct]

words_1 = [x[0] for x in word_freq_1][:25]
values_1 = [int(x[1]) for x in word_freq_1][:25]

words_2 = [x[0] for x in word_freq_2][:25]
values_2 = [int(x[1]) for x in word_freq_2][:25]

print words_2

rects1 = ax.bar(ind, values_1_pct, width,color='r')

rects2 = ax.bar(ind + width, values_2_pct, width, color='y')

# add some text for labels, title and axes ticks
ax.set_ylabel('Words')
ax.set_title('Word Frequencies by Test and Training Set')
ax.set_xticks(ind,minor=False)
ax.set_xticks(ind + width,minor=True)
ax.set_xticklabels(words_1,rotation=90,minor=False,ha='left')
ax.set_xticklabels(words_2,rotation=90,minor=True,ha='left')
ax.tick_params(axis='both', which='major', labelsize=6)
ax.tick_params(axis='both', which='minor', labelsize=6)
vals = ax.get_yticks()
ax.set_yticklabels(['{:1.1f}%'.format(x*100) for x in vals])
fig.tight_layout()

ax.legend((rects1[0], rects2[0]), ('Test', 'Train'))

plt.savefig('test.png')

