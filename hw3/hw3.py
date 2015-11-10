# hw3.py
# ----------
# Ting Zhang

import sys
import re
from collections import Counter
import csv

##define a structure for weights
class myWeight:
    def __int__(self):
        self.weight = None
        self.b = None

##define a structure for one data entry
class oneEntry:
    def __int__(self):
        self.unigram = None
        self.bigram = None
        self.both = None
        self.label = None


##define a structure for word analysis
class DB:
    def __init__(self):
        self.unigram_wd = None
        self.bigram_wd = None
        self.both_wd = None
        self.entry = None

##build unigram feature set from data
def get_unigram(data):
    unigram = []
    myset = set(data)
    unigram = list(myset)

    return unigram

##build bigram feature set from data
def get_bigram(data):
    bigram = []
    i = 0
    while i < len(data)-1:
        bigram = bigram + [data[i] + data[i+1]]
        i = i + 1
    myset = set(bigram)
    bigram = list(myset)

    return bigram

##get feature set contains both unigram and bigram
def get_both(unigram, bigram):
    both = []
    myset = set(unigram + bigram)
    both = list(myset)

    return both

##get word dictionary
def getDB(inF):
    value = 0
    wd={}
    label = []
    thislabel = None
    allEntry = []
        #[[0 for x in range(3)] for x in range(3)]
    allWordsUni = []
    allWordsBi = []
    with open(inF) as f:
        for line in f:
            oneEntryItem = oneEntry()
            data = re.findall(r"[^;,.\s\"]+",line.lower())
            #tokens = re.split(r'[;,."\s]\s*', line)
            for key in data:
                if key == '-':
                    thislabel = '-'
                    oneEntryItem.label=-1
                elif key == '+':
                    thislabel = '+'
                    oneEntryItem.label=1

            data.remove(thislabel)
            #build unigram feature set
            oneEntryItem.unigram = get_unigram(data)
            #build bigram feature set
            oneEntryItem.bigram = get_bigram(data)
            #build both unigram and bigram feature set
            oneEntryItem.both = get_both(oneEntryItem.unigram, oneEntryItem.bigram)

            allEntry = allEntry + [oneEntryItem]

            #build dictionary of both feature sets
            allWordsUni = allWordsUni + oneEntryItem.unigram
            allWordsBi = allWordsBi + oneEntryItem.bigram

    #get distinct words and convert into a dictionary
    myset = set(allWordsUni)
    wdUni = dict((el,0) for el in myset)
    myset = set(allWordsBi)
    wdBi = dict((el,0) for el in myset)
    myset = set(allWordsUni+allWordsBi)
    wdBoth = dict((el,0) for el in myset)
    info = DB()
    info.unigram_wd = wdUni
    info.bigram_wd = wdBi
    info.both_wd = wdBoth
    info.entry = allEntry

    return info

## get a feature of type: 1. unigram, 2.bigram and 3.both
def get_feature(type, data):
    feature = []
    if type == 1:
        feature = data.unigram
    elif type == 2:
        feature = data.bigram
    elif type == 3:
        feature = data.both

    return feature

def parseArgs(args):
  """Parses arguments vector, looking for switches of the form -key {optional value}.
  For example:
	parseArgs([ 'template.py', '-i', '10', '-r', 'l1', '-s', '0.4', '-l', '0.5', '-f', '1' ]) = {'-i':'10', '-r':'l1', '-s:'0.4', '-l':'0.5', '-f':1 }"""
  args_map = {}
  curkey = None
  for i in xrange(1, len(args)):
    if args[i][0] == '-':
      args_map[args[i]] = True
      curkey = args[i]
    else:
      assert curkey
      args_map[curkey] = args[i]
      curkey = None
  return args_map

##predict a single example
def predict_one(w,input_snippet):
    data = input_snippet
    sign = 0
    for i in data:
        if i in w.weight:
            sign = sign + w.weight[i]
    sign = sign + w.b
    return sign

def add_reg(g,lmbd,w,reg):
    #g = g - lmbd * (w.weight^(reg - 1))
    for i in w.weight:
        w.weight[i] = lmbd * w.weight[i]^(reg - 1)
        g[i] = g[i] - w.weight[i]

    return g

def update_weight(w,g,gb,stepSize):
    for i in w.weight:
        w.weight[i] = w.weight[i]+stepSize * g[i]

    w.b = w.b + stepSize * gb

    return w

## Gradient Descent Algorithm
def GD(maxIterations, regularization, stepSize, lmbd, featureSet,data,w):
    if regularization == 'l1':
        reg = 1
    elif regularization == 'l2':
        reg = 2
    else:
        reg = 1

    #data = get_data(featureSet)
    i = 0
    while i < maxIterations:
        # iterate over all training samples
        g = dict.fromkeys(w.weight, 0)
        gb = 0
        for entry in data:
            feature = get_feature(featureSet, entry)
            sign = predict_one(w,feature)
            if sign * entry.label <= 1:
                for one_feature in feature:
                    if one_feature in w.weight:
                        g[one_feature] = g[one_feature] + entry.label
                gb = gb + entry.label

        # add regularization term
        g = add_reg(g,lmbd,w,reg)
        w = update_weight(w,g,gb,stepSize)

        # go to next iteration
        i = i + 1

    return w

##get dictionary for each feature set
def get_wd(type,wd):
    oneWd = {}
    if type == 1:
        oneWd = wd.unigram_wd
    elif type == 2:
        oneWd = wd.bigram_wd
    elif type == 3:
        oneWd = wd.both_wd

    return oneWd

##Compute accuracy
#-------
def get_performance(featureSet, weights,data):
    accuracy = 0
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    for one in data:
        feature = get_feature(featureSet,one)
        sign = predict_one(weights,feature)
        if 1.0*sign*one.label>1:
            accuracy = accuracy + 1
            if one.label == 1:
                true_positive = true_positive + 1
            elif one.label == -1:
                true_negative = true_negative + 1
        else:
            if one.label == 1:
                false_negative = false_negative + 1
            elif one.label == -1:
                false_positive = false_positive + 1

    accuracy = 1.0*accuracy/len(data)
    if (true_positive + false_positive) == 0:
        precision = float('inf')
    else:
        precision = 1.0 * true_positive / (true_positive + false_positive)
    if (true_positive + false_negative) == 0:
        recall = float('inf')
    else:
        recall = 1.0 * true_positive / (true_positive + false_negative)
    f_score = 2.0 * precision * recall / (precision + recall)

    performance = [accuracy, precision, recall, f_score]

    return performance

# main
# ----
# The main program loop
# You should modify this function to run your experiments

def main():
    # Read in the data file and form a database contain the word dictionary, labels and every data entry
    db_tr = getDB("train.csv")
    db_va = getDB("validation.csv")
    db_te = getDB("test.csv")

    # 1. Experiment with Unigram feature set
    # 2. Experiment with Bigram feature set
    # 3. Experiment with both feature set
    numOfExp = 3
    expName = ['Unigram','Bigram','Both']
    expId = 1
    hyperP = 7
    stepSize = 1
    lmbd = 1

    trainP_l1 = [[[]]for x in range(numOfExp)]
    validateP_l1 = [[[]]for x in range(numOfExp)]

    trainP_l2 = [[[]]for x in range(numOfExp)]
    validateP_l2 = [[[]]for x in range(numOfExp)]

    while expId <= numOfExp:
        maxIterations = 1
        wd = get_wd(expId,db_tr)
        weightP = myWeight()
        weightP.weight = wd
        while maxIterations < hyperP:
            print "maxIterations: ", maxIterations
            weightP.weight = dict.fromkeys(weightP.weight, 0)
            weightP.b = 0
            # train with GD of high loss and l1 norm
            weightP = GD(maxIterations,'l1',stepSize,lmbd,expId,db_tr.entry,weightP)

            trainP_l1[expId-1] = trainP_l1[expId-1] + [get_performance(expId,weightP,db_tr.entry)]
            print "Train Accuracy of GD, l1 with ",expName[expId-1],": ", trainP_l1[expId-1][maxIterations]
            validateP_l1[expId-1] = validateP_l1[expId-1] + [get_performance(expId,weightP,db_va.entry)]
            print "Validate Accuracy of GD, l1 with ",expName[expId-1],": ", validateP_l1[expId-1][maxIterations]

            # train with GD of high loss and l2 norm
            weightP.weight = dict.fromkeys(weightP.weight, 0)
            weightP.b = 0
            # train with GD of high loss and l1 norm
            weightP = GD(maxIterations,'l2',stepSize,lmbd,expId,db_tr.entry,weightP)

            trainP_l2[expId-1] = trainP_l2[expId-1] + [get_performance(expId,weightP,db_tr.entry)]
            print "Train Accuracy of GD, l2 with ",expName[expId-1],": ", trainP_l2[expId-1][maxIterations]
            validateP_l2[expId-1] = validateP_l2[expId-1] + [get_performance(expId,weightP,db_va.entry)]
            print "Validate Accuracy of GD, l2 with ",expName[expId-1],": ", validateP_l2[expId-1][maxIterations]

                #if testOn == 1:
                    #if maxIterations == optIter[0][expId-1]:
                        #testErrP = get_err('perceptron',expId,weightP,db_te.entry)
                        #print "Test Accuracy of GD with ",expName[expId-1],": ", testErrP
                        #break
            # tune hyperparameter: max iteration number
            maxIterations = maxIterations +1

        # experiment for next feature set
        expId = expId + 1


if __name__ == '__main__':
    main()
