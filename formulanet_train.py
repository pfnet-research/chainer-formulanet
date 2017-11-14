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
import numpy as np
import os

import formulanet
import holstep
import symbols

def main():
    parser = argparse.ArgumentParser(description='chainer formulanet trainer')

    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='Number of examples in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=5,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--dataset', '-i', default="holstep",
                        help='Directory of holstep repository')
    parser.add_argument('--out', '-o', default='results',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
#    parser.add_argument('--seed', type=int, default=0,
#                        help='Random seed')
#    parser.add_argument('--snapshot_interval', type=int, default=10000,
#                        help='Interval of snapshot')
#    parser.add_argument('--display_interval', type=int, default=100,
#                        help='Interval of displaying log to console')
    parser.add_argument('--conditional', action='store_true', help='Use contional model')
    parser.add_argument('--preserve-order', action='store_true', help='Use order-preserving model')
    parser.add_argument('--steps', type=int, default="3", help='Number of update steps')
    args = parser.parse_args()

    print('# GPU: {}'.format(args.gpu))
    print('# epoch: {}'.format(args.epoch))
    print('')
    print('# conditional: {}'.format(args.conditional))
    print('# order_preserving: {}'.format(args.preserve_order))
    print('# steps: {}'.format(args.steps))
    print('')

    train = []
    for i in range(1,10000):
#    for i in [1]:
        fname = "%s/train/%05d" % (args.dataset, i)
        train.append(holstep.read_file(fname))
    train = formulanet.Dataset(symbols.symbols, train)

    test = []
    for i in range(1,1412):
#    for i in [1]:
        fname = "%s/test/%04d" % (args.dataset, i)
        test.append(holstep.read_file(fname))
    test  = formulanet.Dataset(symbols.symbols, test)
    
    train_iter = iterators.SerialIterator(train, args.batchsize)
    test_iter = iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

    model = formulanet.FormulaNet(vocab_size=len(symbols.symbols), steps=args.steps, order_preserving=args.preserve_order, conditional=args.conditional)
    if args.gpu >= 0:
        model.to_gpu(args.gpu)
    
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    updater = training.StandardUpdater(train_iter, optimizer, converter=formulanet.convert, device=args.gpu)
    
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=os.path.join(args.out))    
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
    trainer.extend(extensions.snapshot_object(model, filename='model_epoch-{.updater.epoch}'))
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu, converter=formulanet.convert))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
    #trainer.extend(extensions.dump_graph('main/loss'))
    
    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)
    
    trainer.run()
    
    chainer.serializers.save_npz(os.path.join(args.out, args.embedding, 'model_final'), model)
    chainer.serializers.save_npz(os.path.join(args.out, args.embedding, 'optimizer_final'), optimizer)

if __name__ == '__main__':
    main()
