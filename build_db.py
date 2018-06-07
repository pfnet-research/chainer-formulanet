import argparse
import sys
from pathlib import Path

import h5py

import formulanet
import symbols

sys.setrecursionlimit(10000)

parser = argparse.ArgumentParser(description='chainer formulanet database builder')
parser.add_argument('--out', '-o', default='results',
                    help='directory to output results')
parser.add_argument('--holstep-dir', default='holstep',
                    help='holstep dataset directory')
args = parser.parse_args()

Path(args.out).mkdir(exist_ok=True, parents=True)

print("converting train data files ..")
with h5py.File(str(Path(args.out) / "train.h5"), 'w') as h5f:
    ds = formulanet.Dataset(symbols.symbols, h5f)
    ds.init_db()
    for i in range(1, 10000):
        fname = Path(args.holstep_dir) / "train" / ("%05d" % i)
        print("loading %s" % fname)
        ds.add_file("%05d" % i, fname)

print("converting test data files ..")
with h5py.File(str(Path(args.out) / "test.h5"), 'w') as h5f:
    ds = formulanet.Dataset(symbols.symbols, h5f)
    ds.init_db()
    for i in range(1, 1412):
        fname = Path(args.holstep_dir) / "test" / ("%04d" % i)
        print("loading %s" % fname)
        ds.add_file("%04d" % i, fname)
