# -*- coding: utf-8 -*-

# to avoid "_tkinter.TclError: no display name and no $DISPLAY environment variable" error
import matplotlib as mpl
mpl.use('Agg')

import argparse
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import iterators
from chainer import optimizers
from chainer import reporter
from chainer import training
from chainer.training import extensions
import h5py
import numpy as np
import os
import pandas as pd
import sys
from tqdm import tqdm

import formulanet
import holstep
import symbols

sys.setrecursionlimit(10000)

def main():
    parser = argparse.ArgumentParser(description='chainer formulanet test')

    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='Number of examples in each mini-batch')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='Set GPU device number.'
                        '(negative value indicates CPU)')
    parser.add_argument('--dataset', '-i', default="holstep",
                        help='HDF5 file')
    parser.add_argument('--out', '-o',
                        help='output CSV file')
    parser.add_argument('--model', '-m', default='',
                        help='Saved model file')
    parser.add_argument('--conditional', action='store_true', help='Use contional model')
    parser.add_argument('--preserve-order', action='store_true', help='Use order-preserving model')
    parser.add_argument('--steps', type=int, default="3", help='Number of update steps')

    args = parser.parse_args()

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()

    print('# GPU: {}'.format(args.gpu))
    print('# conditional: {}'.format(args.conditional))
    print('# order_preserving: {}'.format(args.preserve_order))
    print('# steps: {}'.format(args.steps))
    print('')

    test_h5f = h5py.File(args.dataset,  'r')
    test = formulanet.Dataset(symbols.symbols, test_h5f)
    test_iter = iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)
    print(len(test))

    model = formulanet.FormulaNet(vocab_size=len(symbols.symbols), steps=args.steps, order_preserving=args.preserve_order, conditional=args.conditional)
    chainer.serializers.load_npz(args.model, model)
    if args.gpu >= 0:
        model.to_gpu()

    with chainer.using_config('train', False):
        with chainer.using_config('enable_backprop', False):
            expected = []
            logits = []

            with tqdm(total=len(test)) as pbar:
                for batch in test_iter:
                    gs, tuples = formulanet.convert(batch, args.gpu)
                    logits1, _loss = model._forward(gs, tuples)
                    logits.append(chainer.cuda.to_cpu(logits1.data))
                    expected.extend(1 if y else 0 for (conj, stmt, y) in tuples)
                    pbar.update(len(batch))

            logits = np.concatenate(logits)
            expected = np.array(expected, dtype=np.int32)

            df = pd.DataFrame({"logits_false": logits[:,0], "logits_true": logits[:,1], "expected": expected})
            df.to_csv(args.out, index=False)

            accuracy = F.accuracy(logits, expected).data
            precision, recall, F_beta_score, support = F.classification_summary(logits, expected)
            print("accuracy: {}".format(accuracy))
            print("precision: {}".format(precision.data[1]))
            print("recall: {}".format(recall.data[1]))
            print("F beta score: {}".format(F_beta_score.data[1]))
            print("support: {}".format(support.data))

if __name__ == '__main__':
    main()
