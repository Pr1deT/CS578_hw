# main.py
# -------
# YOUR NAME HERE

##Classify
#---------

def classify(decisionTree, example):
    pass
    return results


##Learn
#-------
def learn(dataset):
    pass
    return learner.dt

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
    arguments = validateInput(sys.argv)
    pruneFlag, valSetSize, maxDepth = arguments
    print pruneFlag, valSetSize, maxDepth

    # Read in the data file


    f = open("data.csv")


    # ====================================
    # WRITE CODE FOR YOUR EXPERIMENTS HERE
    # ====================================

if __name__ == '__main__':
    main()



