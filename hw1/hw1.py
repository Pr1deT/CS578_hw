# hw1.py
# -------
# Ting Zhang
# zhan1013@purdue.edu

import sys
import math
import numpy as np
import operator

##Define a class for DT
# ---------
class DT:
    def __init__(self):
        self.parent = None
        self.parentValue = None
        self.child = []
        self.feature = None
        self.level = None
        self.label = None

    def copy(self,tree):
        self.parent = tree.parent
        self.parentValue = tree.parentValue
        self.feature = tree.feature
        self.level = tree.level
        self.label = tree.label
        self.child = []
        if tree.child :
            for child in tree.child:
                schild = DT()
                schild.copy(child)
                self.child = self.child + [schild]
        else:
            return self

    def addchild(self,parentf, value):
        child = DT()
        child.parent = parentf
        child.parentValue = value
        child.feature = None
        child.level = self.level + 1
        child.label = None
        if self.child == None :
            self.child = child
        else:
            self.child = self.child + [child]
        return child



##Classify
# ---------

def classify(decisionTree, example):
    results = -1
    #node = decisionTree
    node = DT()
    node.copy(decisionTree)
    if not node.child:
        results = node.label
    else:
        feature = node.feature
        #print feature
        for child in node.child:
            if child.parentValue == example[feature]:
                return classify(child,example)
        return node.label

    return results


## Get the information gain of a feature
# --------
def getGain(dataset, feature):
    fcount = np.array([0]*10)
    H = np.array([0]*10)
    for i in xrange(10):
        subset = dataset[np.where(dataset[:,feature]==(i+1))]
        if len(subset)==0 :
            continue
        else:
            benign = 1.0 * (subset[:,9]==0).sum() / len(subset)
            malignant = 1.0 * (subset[:,9]==1).sum() / len(subset)
            fcount[i] = len(subset)
            if (benign==0.0) | (malignant==0.0) :
                H[i]=0.0
            else:
                H[i] = -1.0 * benign * math.log(benign, 2) - 1.0 * malignant * math.log(malignant, 2)
    fcount = 1.0 * fcount / len(dataset)

    # get entropy
    entropy = sum(fcount * H)

    # get entropy of the whole dataset
    benign = 1.0 * (dataset[:,9]==0).sum() / len(dataset)
    malignant = 1.0 * (dataset[:,9]==1).sum() / len(dataset)
    fEntropy = -1.0 * benign * math.log(benign) - 1.0 * malignant * math.log(malignant)

    # get information gain
    gain = fEntropy - entropy

    return gain


## Get the feature with highest information gain
# --------
def getFeature(dataset, fset):
    feature = None
    gainMax = -float("inf")
    for i in xrange(len(fset)):
        if fset[i] == 1:
            gain = getGain(dataset, i)
            if gain > gainMax:
                gainMax = gain
                feature = i

    return feature


##Learn
# -------
def learn(node, dataset, fset, maxD=+float("inf")):
    if node==None :
        node = DT()
        node.level = 0

    # get num of instances of different classes
    benign = 1.0 * (dataset[:,9]==0).sum() / len(dataset)
    malignant = 1.0 * (dataset[:,9]==1).sum() / len(dataset)

    # assign majority votes
    node.label = 0 if benign > malignant else 1

    #check if all data has same label
    if benign==1.0:
        node.label = 0
    elif benign==0.0:
        node.label = 1
    # check if the remaining feature set is empty
    elif (np.count_nonzero(fset) == 0)|(node.level == (maxD-1)):
        node.label = 0 if benign > malignant else 1
    else:
        feature = getFeature(dataset,fset)
        node.feature = feature
        # update feature set
        fset[feature] = 0
        keepset = list(fset)
        for i in xrange(10) :
            #print "feature ", feature," = ", i+1,"; fset: ",fset, " len of dataset:", len(dataset)
            # update dataset
            currentdataset = dataset[np.where(dataset[:,feature]==(i+1))]
            #keepdataset = np.copy(currentdataset)
            if len(currentdataset)==0 :
                continue
            child = node.addchild(node.feature,i+1)
            # learn for the new feature
            learn(child,currentdataset,keepset,maxD)

    return node


##Compute the depth of a tree
# -------
def getDepth(decisionTree):
    if not decisionTree.child:
        return 1
    else:
        d = []
        for child in decisionTree.child:
            d = d + [getDepth(child)]
        dMax = max(d)
        dMax = dMax +1
        return dMax


##Post Prune
# -------
def substitute(decisionTree,path,node):
    if decisionTree.feature in path:
        for child in decisionTree.child:
            if child.parentValue==path[decisionTree.feature]:
                substitute(child,path,node)
                continue
    else:
        decisionTree.copy(node)
    return decisionTree

def getLabel(path,dataset):

    # get num of instances of different classes
    benign = 1.0 * (dataset[:,9]==0).sum() / len(dataset)
    malignant = 1.0 * (dataset[:,9]==1).sum() / len(dataset)

    for key,value in path.iteritems():
        dataset = dataset[np.where(dataset[:,key]==value)]

    if len(dataset)==0 :
        label = 0 if benign > malignant else 1
        return label

    # get num of instances of different classes
    benign = 1.0 * (dataset[:,9]==0).sum() / len(dataset)
    malignant = 1.0 * (dataset[:,9]==1).sum() / len(dataset)

    label = 0 if benign > malignant else 1

    return label

def prune(decisionTree,dataset):
    depth = getDepth(decisionTree)
    node = DT()
    node.copy(decisionTree)
    thislevel = 0
    path = {}
    while thislevel < depth-1:
        childDepth = {}
        for child in node.child:
            childDepth[child] = getDepth(child)
        childM = max(childDepth.iteritems(), key=operator.itemgetter(1))[0]
        path[node.feature] = childM.parentValue
        node.copy(childM)
        thislevel = node.level+1
    #print thislevel
    node.child = []
    label = getLabel(path,dataset)
    #node.feature = label
    node.label = label
    # substitute in original decision tree
    decisionTree = substitute(decisionTree,path,node)

    return decisionTree

def postprune(decisionTree,dataset_v,dataset_tr,accuracyMax):
    accuracy = accuracyMax
    # save the original full tree
    tree = DT()
    tree.copy(decisionTree)
    node = DT()
    while accuracy >= accuracyMax:
        accuracyMax = accuracy
        # save the last worked tree
        #node = DT()
        node.copy(tree)
        #accuracy = test(node,dataset_v,"pruned")
        # get new pruned tree
        tree = prune(tree,dataset_tr)
        print "new tree depth: ", getDepth(tree)
        accuracy = test(tree,dataset_v,"pruning")
        print "\n"
    return node


##Test
# -------
def test(decisionTree,dataset,printName):
    hit = 0
    for data in dataset:
        label = classify(decisionTree,data)
        #print label, data[9]
        if label==data[9]:
            hit = hit + 1
    accuracy = 1.0 * hit / dataset.shape[0]
    print printName,": ",accuracy

    return accuracy


# main
# ----
# The main program loop
# You should modify this function to run your experiments

def parseArgs(args):
    """Parses arguments vector, looking for switches of the form -key {optional value}.
  For example:
    parseArgs([ 'main.py', '-p', 5 ]) = {'-p':5 }"""
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
    valSetSize = 0
    pruneFlag = False
    maxDepth = -1
    if '-p' in args_map:
        pruneFlag = True
        valSetSize = int(args_map['-p'])
    if '-d' in args_map:
        maxDepth = int(args_map['-d'])
    return [pruneFlag, valSetSize, maxDepth]


def main():
    #arguments = validateInput(sys.argv)
    #pruneFlag, valSetSize, maxDepth = arguments
    #print pruneFlag, valSetSize, maxDepth

    # Read in the data file
    f_tr = open("training.csv")
    f_v = open("validating.csv")
    f_te = open("testing.csv")

    data_tr = np.genfromtxt(f_tr) #, converters={"?": 0})
    data_v = np.genfromtxt(f_v) #, converters={"?": 0})
    data_te = np.genfromtxt(f_te)
    # ====================================
    # WRITE CODE FOR YOUR EXPERIMENTS HERE
    # ====================================


    # Basic ID3
    fset = [1]*8
    root = learn(None,data_tr,fset,10)
    print "Basic ID3:"
    test(root,data_tr,"training")
    test(root,data_v,"validating")
    test(root,data_te,"testing")
    print "\n"

    # Fixed Depth
    print "Fixed Depth: "
    for i in range(1,9):
        fset = [1]*8
        root = learn(None,data_tr,fset,i)
        depth = getDepth(root)
        # print "i: ",i, ", depth: ", depth
        # print "feature at root: ", root.feature
        # print "feature at level 1: "
        # for child in root.child:
        #     print child.feature
        print "Depth: ", depth
        test(root,data_tr,"training")
        accuracy_v = test(root,data_v,"validating")
        #test(root,data_te,"testing")
        print "\n"
    print "\n"

    # Pruned Tree
    print "Pruning Tree: "
    prunedTree = postprune(root,data_v,data_tr,accuracy_v)
    print "Pruning finished!"
    test(prunedTree,data_tr,"training")
    test(prunedTree,data_v,"validating")
    test(prunedTree,data_te,"testing")

if __name__ == '__main__':
    main()
