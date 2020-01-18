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
import sys

import formulanet
import holstep
import symbols

sys.setrecursionlimit(10000)


def main():
    parser = argparse.ArgumentParser(description='chainer formulanet trainer')

    parser.add_argument('--chainermn', action='store_true', help='Use ChainerMN')
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='Number of examples in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=5,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--devices', type=str, default='',
                        help='Comma-separated list of devices specifier. '
                        'Either ChainerX device  specifier or an integer. '
                        'If non-negative integer, CuPy arrays with specified '
                        'device id are used. If negative integer, NumPy '
                        'arrays are used')
    parser.add_argument('--dataset', '-i', default="holstep",
                        help='Directory of holstep repository')
    parser.add_argument('--out', '-o', default='results',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--autoload', action='store_true',
                        help='Automatically load trainer snapshots in case'
                        ' of preemption or other temporary system failure')
    #    parser.add_argument('--seed', type=int, default=0,
    #                        help='Random seed')
    #    parser.add_argument('--snapshot_interval', type=int, default=10000,
    #                        help='Interval of snapshot')
    #    parser.add_argument('--display_interval', type=int, default=100,
    #                        help='Interval of displaying log to console')
    parser.add_argument('--conditional', action='store_true', help='Use contional model')
    parser.add_argument('--preserve-order', action='store_true', help='Use order-preserving model')
    parser.add_argument('--steps', type=int, default="3", help='Number of update steps')

    parser.add_argument('--run-id', type=str, default="formulanet_train",
                        help='ID of the task name')
    parser.add_argument('--checkpointer-path', type=str, default=None,
                        help='Path for chainermn.create_multi_node_checkpointer')

    args = parser.parse_args()

    if args.chainermn:
        # matplotlib.font_manager should be imported before mpi4py.MPI
        # to avoid MPI issue with fork() system call.
        import matplotlib.font_manager
        import chainermn
        from chainermn.extensions import create_multi_node_checkpointer
        comm = chainermn.create_communicator()
        devices = [chainer.get_device("@cupy:" + str(comm.intra_rank))]
    else:
        if args.devices == '':
            devices = [chainer.get_device(-1)]
        else:
            devices = list(map(chainer.get_device, args.devices.split(',')))
        print('# Devices: {}'.format(",".join(map(str, devices))))
    devices[0].use()

    if not args.chainermn or comm.rank == 0:
        print('# epoch: {}'.format(args.epoch))
        print('# conditional: {}'.format(args.conditional))
        print('# order_preserving: {}'.format(args.preserve_order))
        print('# steps: {}'.format(args.steps))
        print('')

    train_h5f = h5py.File(os.path.join(args.dataset, "train.h5"), 'r')
    test_h5f = h5py.File(os.path.join(args.dataset, "test.h5"), 'r')

    if not args.chainermn or comm.rank == 0:
        train = formulanet.Dataset(symbols.symbols, train_h5f)
        test = formulanet.Dataset(symbols.symbols, test_h5f)
    else:
        train, test = None, None

    if args.chainermn:
        # XXX: h5py.File cannot be distributed
        if comm.rank == 0:
            train._h5f = None
            test._h5f = None
        train = chainermn.scatter_dataset(train, comm)
        test = chainermn.scatter_dataset(test, comm)
        # We assume train and test are chainer.datasets.SubDataset.
        train._dataset._h5f = train_h5f
        test._dataset._h5f = test_h5f

    train_iter = iterators.SerialIterator(train, args.batchsize)
    test_iter = iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

    model = formulanet.FormulaNet(vocab_size=len(symbols.symbols), steps=args.steps,
                                  order_preserving=args.preserve_order, conditional=args.conditional)
    if len(devices) == 1:
        model.to_device(devices[0])

    # "We train our networks using RMSProp [47] with 0.001 learning rate and 1 × 10−4 weight decay.
    # We lower the learning rate by 3X after each epoch."
    if chainer.get_dtype() == np.float16:
        optimizer = optimizers.RMSprop(lr=0.001, eps=1e-07)
    else:
        optimizer = optimizers.RMSprop(lr=0.001)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(10 ** (-4)))
    if args.chainermn:
        optimizer = chainermn.create_multi_node_optimizer(optimizer, comm)

    if len(devices) == 1:
        updater = training.updaters.StandardUpdater(train_iter, optimizer, converter=formulanet.convert, device=devices[0])
    else:
        devices_dict = {}
        devices_dict["main"] = devices[0]
        for i in range(1, len(devices)):
            devices_dict["device" + str(i)] = devices[i]
        updater = training.updaters.ParallelUpdater(train_iter, optimizer, converter=formulanet.convert, devices=devices_dict)

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=os.path.join(args.out))
    trainer.extend(extensions.ExponentialShift("lr", rate=1 / 3.0), trigger=(1, 'epoch'))

    evaluator = extensions.Evaluator(test_iter, model, device=devices[0], converter=formulanet.convert)
    if args.chainermn:
        evaluator = chainermn.create_multi_node_evaluator(evaluator, comm)
    trainer.extend(evaluator)

    if not args.chainermn or comm.rank == 0:
        trainer.extend(extensions.LogReport())
        trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy',
             'elapsed_time']))
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
        trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch',
                                             file_name='accuracy.png'))
        trainer.extend(extensions.ProgressBar(update_interval=10))
        trainer.extend(extensions.snapshot_object(model, filename='model_epoch-{.updater.epoch}'))

    snapshot = extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}', autoload=args.autoload)
    if args.chainermn:
        replica_sets = [[0], range(1, comm.size)]
        snapshot = chainermn.extensions.multi_node_snapshot(comm, snapshot, replica_sets)
    trainer.extend(snapshot, trigger=(1, 'epoch'))

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()

    if not args.chainermn or comm.rank == 0:
        chainer.serializers.save_npz(os.path.join(args.out, 'model_final'), model)
        chainer.serializers.save_npz(os.path.join(args.out, 'optimizer_final'), optimizer)


if __name__ == '__main__':
    main()
