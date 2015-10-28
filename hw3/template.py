import sys

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

def validateInput(args):
    args_map = parseArgs(args)

    maxIterations = 10 # the maximum number of iterations. should be a positive integer
	regularization = 'l1' # 'l1' or 'l2'
	stepSize = 0.1 # 0 < stepSize <= 1
	lmbd = 0.1 # 0 < lmbd <= 1
	featureSet = 1 # 1: original attribute, 2: pairs of attributes, 3: both

    if '-i' in args_map:
      maxIterations = int(args_map['-i'])
    if '-r' in args_map:
      regularization = args_map['-r']
    if '-s' in args_map:
      stepSize = float(args_map['-s'])
    if '-l' in args_map:
      lmbd = float(args_map['-l'])
    if '-f' in args_map:
      featureSet = int(args_map['-f'])

    assert maxIterations > 0
    assert regularization in ['l1', 'l2']
    assert stepSize > 0 and stepSize <= 1
    assert lmbd > 0 and lmbd <= 1
    assert featureSet in [1, 2, 3]
	
    return [maxIterations, regularization, stepSize, lmbd, featureSet]

## Gradient Descent Algorithm
def GD(maxIterations, regularization, stepSize, lmbd, featureSet):
	pass

# main
# ----
# The main program loop
# You should modify this function to run your experiments

def main():
    arguments = validateInput(sys.argv)
    maxIterations, regularization, stepSize, lmbd, featureSet = arguments
    print maxIterations, regularization, stepSize, lmbd, featureSet

    # ====================================
    # WRITE CODE FOR YOUR EXPERIMENTS HERE
    # ====================================

if __name__ == '__main__':
    main()
