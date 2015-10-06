import sys
import re
from collections import Counter
import csv

# hw2.py
# -------
# Ting Zhang

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

##Perceptron--------------------------------------------------
##predict a single example
def predict_one_p(weights,input_snippet):
    data = input_snippet
    sign = 0
    for i in data:
        if i in weights.weight:
            sign = sign + weights.weight[i]
    sign = sign + weights.b
    return sign

##update weight
def update_weight_p(weights,entry,label):
    data = entry
    for i in data:
        if i in weights.weight:
            weights.weight[i] = weights.weight[i]+label
    weights.b = weights.b+label

    return weights

##Perceptron
#-----------
def perceptron(maxIterations, featureSet, weights, data):
    currentIter = 0
    while currentIter < maxIterations:
        for entry in data:
            feature = get_feature(featureSet, entry)
            sign = predict_one_p(weights,feature)
            if sign*entry.label <= 0: #update when there is an error
                weights = update_weight_p(weights,feature,entry.label)
        currentIter +=1

    return weights 

##Winnow-----------------------------------------------------------
##predict a single example
def predict_one_w(weights,input_snippet):
    data = input_snippet
    theta = weights.b
    sign = 0
    for i in data:
        if i in weights.weight:
            sign = sign + weights.weight[i]
    if sign < theta:
        sign = -1
    else:
        sign = 1
    return sign

##update weight
def update_weight_w(weights,entry,sign):
    data = entry
    if sign == 1:
        for i in data:
            if i in weights.weight:
                weights.weight[i] = 2.0 * weights.weight[i]
    elif sign == -1:
        for i in data:
            if i in weights.weight:
                weights.weight[i] = 1.0 * weights.weight[i] / 2.0

    return weights

##Winnow
#-------
def winnow(maxIterations, featureSet, weights, data):
    currentIter = 0
    while currentIter < maxIterations:
        for entry in data:
            feature = get_feature(featureSet,entry)
            sign = predict_one_w(weights,feature)
            if 1.0*sign*entry.label <= 0: #update when there is an error
                weights = update_weight_w(weights,feature,entry.label)
        currentIter  = currentIter + 1

    return weights 

##Compute accuracy
#-------
def get_err(algorithm,featureSet, weights,data):
    accuracy = 0
    if algorithm=='perceptron':
        for one in data:
            feature = get_feature(featureSet,one)
            sign = predict_one_p(weights,feature)
            if 1.0*sign*one.label>0:
                accuracy = accuracy + 1
        accuracy = 1.0*accuracy/len(data)
    elif algorithm == 'winnow':
        for one in data:
            feature = get_feature(featureSet,one)
            sign = predict_one_w(weights,feature)
            if 1.0*sign*one.label>0:
                accuracy = accuracy + 1
        accuracy = 1.0*accuracy/len(data)

    return accuracy

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

# main
# ----
# The main program loop
# You should modify this function to run your experiments

def parseArgs(args):
  """Parses arguments vector, looking for switches of the form -key {optional value}.
  For example:
	parseArgs([ 'hw2.py', '-a', 1, '-i', 10, '-f', 1 ]) = {'-t':1, '-i':10, '-f':1 }"""
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

def validateInput(args):
    args_map = parseArgs(args)

    algorithm = 1 # 1: perceptron, 2: winnow
    maxIterations = 10 # the maximum number of iterations. should be a positive integer
    featureSet = 1 # 1: original attribute, 2: pairs of attributes, 3: both

    if '-a' in args_map:
      algorithm = int(args_map['-a'])
    if '-i' in args_map:
      maxIterations = int(args_map['-i'])
    if '-f' in args_map:
      featureSet = int(args_map['-f'])

    assert algorithm in [1, 2]
    assert maxIterations > 0
    assert featureSet in [1, 2, 3]
	
    return [algorithm, maxIterations, featureSet]

def main():
    #arguments = validateInput(sys.argv)
    #algorithm, maxIterations, featureSet = arguments
    #print algorithm, maxIterations, featureSet

    # ====================================
    # WRITE CODE FOR YOUR EXPERIMENTS HERE
    # ====================================

    # control flag: p==1: run perceptron; w==1: run winnow; testOn==1: run training with test (known number of iterations);
    # testOn==0: run training without test (get number of iterations)
    p = 1
    w = 1
    testOn = 1

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
    hyperP = 100
    optIter = [[40,10,5],[51,25,9]]

    # Perceptron
    if p == 1:
        print "Perceptron--------------------"
        trainErrP = [[] for x in range(numOfExp)]
        validateErrP = [[] for x in range(numOfExp)]
        testErrP = None #[[] for x in range(numOfExp)]

        while expId <= numOfExp:
            maxIterations = 1
            wd = get_wd(expId,db_tr)
            weightP = myWeight()
            weightP.weight = wd
            while maxIterations < hyperP:
                print "maxIterations: ", maxIterations
                weightP.weight = dict.fromkeys(weightP.weight, 0)
                weightP.b = 0
                # train with Perceptron
                weightP = perceptron(maxIterations,expId,weightP,db_tr.entry)

                trainErrP[expId-1] = trainErrP[expId-1] + [get_err('perceptron',expId,weightP,db_tr.entry)]
                print "Train Accuracy of Perceptron with ",expName[expId-1],": ", trainErrP[expId-1][maxIterations-1]

                validateErrP[expId-1] = validateErrP[expId-1] + [get_err('perceptron',expId,weightP,db_va.entry)]
                print "Validate Accuracy of Perceptron with ",expName[expId-1],": ", validateErrP[expId-1][maxIterations-1]

                if testOn == 1:
                    if maxIterations == optIter[0][expId-1]:
                        testErrP = get_err('perceptron',expId,weightP,db_te.entry)
                        print "Test Accuracy of Perceptron with ",expName[expId-1],": ", testErrP
                        break

                if maxIterations > 1 and trainErrP[expId-1][maxIterations-1] == trainErrP[expId-1][maxIterations-2]\
                        and trainErrP[expId-1][maxIterations-1] == 1.0:
                    break
                # tune hyperparameter: max iteration number
                maxIterations = maxIterations +1

            # experiment for next feature set
            expId = expId + 1

        # Saving the results:
        perceptronValidatefile = "perceptron_validate.csv"
        perceptronTrainfile = "perceptron_train.csv"
        with open(perceptronValidatefile, "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerows(validateErrP)
        with open(perceptronTrainfile, "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerows(trainErrP)

    # Winnow
    if w == 1:
        expId = 1
        print "Winnow--------------------------"
        trainErrW = [[] for x in range(numOfExp)]
        validateErrW = [[] for x in range(numOfExp)]
        testErrW = [] #[[] for x in range(numOfExp)]
        while expId <= numOfExp:
            maxIterations = 1
            wd = get_wd(expId,db_tr)
            wd = dict.fromkeys(wd, 1)
            theta = len(wd)
            weightW = myWeight()
            weightW.weight = wd
            weightW.b = theta
            while maxIterations < hyperP:
                print "maxIterations: ", maxIterations
                # train with Winnow
                weightW.weight = dict.fromkeys(weightW.weight, 1)
                weightW = winnow(maxIterations,expId,weightW,db_tr.entry)

                # train accuracy of winnow
                trainErrW[expId-1] = trainErrW[expId-1] + [get_err('winnow',expId,weightW,db_tr.entry)]
                print "Train Accuracy of Winnow with ",expName[expId-1],": ", trainErrW[expId-1][maxIterations-1]

                # validate accuracy of winnow
                validateErrW[expId-1] = validateErrW[expId-1] + [get_err('winnow',expId,weightW,db_va.entry)]
                print "Validate Accuracy of Winnow with ",expName[expId-1],": ", validateErrW[expId-1][maxIterations-1]

                if testOn == 1:
                    if maxIterations == optIter[1][expId-1]:
                        testErrW = get_err('winnow',expId,weightW,db_te.entry)
                        print "Test Accuracy of Winnow with ",expName[expId-1],": ", testErrW
                        break

                if maxIterations > 1 and trainErrW[expId-1][maxIterations-1] == trainErrW[expId-1][maxIterations-2]\
                        and trainErrW[expId-1][maxIterations-1] == 1.0:
                    break
                # tune hyperparameter: max iteration number
                maxIterations = maxIterations +1

            # experiment for next feature set
            expId = expId + 1
        # Saving the results:
        winnowValidatefile = "winnow_validate.csv"
        winnowTrainfile = "winnow_train.csv"
        with open(winnowValidatefile, "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerows(validateErrW)
        with open(winnowTrainfile, "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerows(trainErrW)

    # Getting back the objects:
    #with open('objs.pickle') as f:
        #obj0, obj1, obj2 = pickle.load(f)


if __name__ == '__main__':
    main()
