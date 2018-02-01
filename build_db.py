import pickle
import os
import sys
import h5py

import formulanet
import holstep
import symbols

sys.setrecursionlimit(10000)

os.makedirs("results", exist_ok=True)

print("converting train data files ..")
with h5py.File("results/train.h5", 'w') as h5f:
    ds = formulanet.Dataset(symbols.symbols, h5f)
    ds.init_db()
    for i in range(1,10000):
    #for i in [1]:
        fname = "holstep/train/%05d" % i
        print("loading %s" % fname)
        ds.add_file("%05d" % i, fname)

print("converting test data files ..")
with h5py.File("results/test.h5", 'w') as h5f:
    ds = formulanet.Dataset(symbols.symbols, h5f)
    ds.init_db()
    for i in range(1,1412):
    #for i in [1]:
        fname = "holstep/test/%04d" % i
        print("loading %s" % fname)
        ds.add_file("%04d" % i, fname)
