import pickle
import os
import sys
import gzip

import formulanet
import holstep
import symbols

sys.setrecursionlimit(10000)

os.makedirs("results/train", exist_ok=True)
os.makedirs("results/test", exist_ok=True)

with gzip.open('results/symbols.pkl.gz', mode='wb') as f:
    pickle.dump(symbols.symbols, f)

print("converting train data files ..")
for i in range(1,10000):
    fname = "holstep/train/%05d" % i
    print("loading %s" % fname)
    xs = formulanet.Dataset(symbols.symbols, [holstep.read_file(fname)])
    with gzip.open('results/train/%05d.pkl.gz' % i, mode='wb') as f:
        pickle.dump(xs._examples, f)

print("converting test data files ..")
xs = []
for i in range(1,1412):
    fname = "holstep/test/%04d" % i
    print("loading %s" % fname)
    xs = formulanet.Dataset(symbols.symbols, [holstep.read_file(fname)])
    with gzip.open('results/test/%04d.pkl.gz' % i, mode='wb') as f:
        pickle.dump(xs._examples, f)
