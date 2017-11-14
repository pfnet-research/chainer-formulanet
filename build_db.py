import pickle
import sys

import formulanet
import holstep
import symbols

sys.setrecursionlimit(10000)

print("building results/train.pickle ..")
xs = []
for i in range(1,9300):
    fname = "holstep/train/%05d" % i
    print("loading %s" % fname)
    xs.append(holstep.read_file(fname))
xs = formulanet.Dataset(symbols.symbols, xs)
with open('results/train.pickle', mode='wb') as f:
    pickle.dump(xs, f)

print("building results/val.pickle ..")
xs = []
for i in range(9300,10000):
    fname = "holstep/train/%05d" % i
    print("loading %s" % fname)
    xs.append(holstep.read_file(fname))
xs = formulanet.Dataset(symbols.symbols, xs)
with open('results/val.pickle', mode='wb') as f:
    pickle.dump(xs, f)

print("building results/test.pickle ..")
xs = []
for i in range(1,1412):
    fname = "holstep/test/%04d" % i
    print("loading %s" % fname)
    xs.append(holstep.read_file(fname))
xs = formulanet.Dataset(symbols.symbols, xs)
with open('results/test.pickle', mode='wb') as f:
    pickle.dump(xs, f)
