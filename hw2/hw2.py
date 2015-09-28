import sys
import re
from collections import Counter

# hw2.py
# -------
# Ting Zhang


##define a structure for word analysis
class DB:
    def __init__(self):
        self.wd = None
        self.label = None
        self.entry = None

##get word dictionary
def getDB(inF):
    value = 0
    wd={}
    label = []
    thislabel = None
    entry = []
    with open(inF) as f:
        for line in f:
            oneEntry = None
            oneEntry = Counter(re.findall(r"[^;,.\s\"]+",line.lower()))
            #tokens = re.split(r'[;,."\s]\s*', line)
            for key in oneEntry:
                if key == '-':
                    thislabel = -1
                    label = label+[thislabel]
                elif key == '+':
                    thislabel = 1
                    label = label+[thislabel]
                elif key in wd:
                    continue
                else:
                    wd[key]=value
            del oneEntry[thislabel]
            #entry[oneEntry] = thislabel
            entry = entry + [oneEntry]

    info = DB()
    info.wd = wd
    info.label = label
    info.entry = entry

    return info


##predict a single example
def predict_one(weights, input_snippet):
	pass
	return sign

##update weight
def update_weight(weights):
    pass
    return weights

##Perceptron
#-----------
def perceptron(maxIterations, featureSet, weights):
    currentIter = 0
    while currentIter < maxIterations:
        for entry in featureSet:
            sign = predict_one(weights,entry)
            if sign*featureSet[entry] <= 0: #update when there is an error
                weights = update_weight(weights)
        currentIter +=1

    return weights 


##Winnow
#-------
def winnow(maxIterations, featureSet):
    pass
    return weights 

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

    # Read in the data file and form a database contain the word dictionary, labels and every data entry
    db_tr = getDB("train.csv")
    wd = db_tr.wd

    print data_tr
    #f_tr = open("train.csv")
    #f_v = open("validation.csv")
    #f_te = open("test.csv")


if __name__ == '__main__':
    main()
