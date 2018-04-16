# Chainer implementation of Formulanet

This is an experimental Chainer implementation of FormulaNet [1], a graph embedding based
premise selection method for theorem proving.

Disclaimer: PFN provides no warranty or support for this implementation. Use it at your own risk.

## Requirement

* Python 3
* [Chainer](https://chainer.org/) >= 3.3.0
* [funcparserlib](https://pypi.python.org/pypi/funcparserlib/)

## Usage

### Dataset preparation

```
$ wget http://cl-informatik.uibk.ac.at/cek/holstep/holstep.tgz
$ tar zxf holstep.tgz
$ python build_db.py -o db
```

### Training

```
$ python formulanet_train.py --dataset db
```

`formulanet_train.py` has several options for configuring models.

* `--conditional`: Use contional model. Default is unconditional model.
* `--preserve-order`: Use order-preserving model (i.e. *FormulaNet* in the original paper). Default is without order information (i.e. *FormulaNet-basic* in the original paper).
* `--steps STEPS`: Number of update steps

### Testing

Pretrained model `formulanet-basic-unconditional-3steps.npz` is included in this repository.

```
$ python formulanet_test.py --model formulanet-basic-unconditional-3steps.npz --dataset db/test.h5 --gpu 0
...
accuracy: 0.8891751170158386
precision: 0.9018562609300268
recall: 0.8733969290414733
F beta score: 0.887398477223135
support: [98015 98015]
```

## Results

Observed accurarcy in our experiment is similar but somewhat lower compared to the original paper.

### Classification accuracy on the test set of our approach versus baseline methods on HolStep

[1, Table 1] + our results:

|  | CNN | CNN-LSTM | Formulanet-basic (orig) | FormulaNet (orig) | Formulanet-basic (ours) | Formulanet (ours) |
|:-|:---:|:--------:|:-----------------------:|:-----------------:|:-----------------------:|:-----------------:|
|Unconditional | 83 | 83 | 89.0 | 90.0 | 89.9 | 89.9
|Conditional   | 82 | 83 | 89.1 | 90.3 | 89.4 | 89.8

### Classification accuracy with different numbers of update steps on conditional premise selection.

Results reported in the paper ([1, Table 3]):

|Number of steps  | 0  | 1  | 2  | 3  | 4  |
|:----------------|:--:|:--:|:--:|:--:|:--:|
|FormulaNet-basic |81.5|89.3|89.8|89.9|90.0|
|FormulaNet       |81.5|90.4|91.0|91.1|90.8|

Results of our experiment:

|Number of steps  | 0  | 1  | 2  | 3  | 4  |
|:----------------|:--:|:--:|:--:|:--:|:--:|
|FormulaNet-basic |74.2|87.7|89.1|89.2|89.4|
|FormulaNet       |74.2|89.0|89.8|89.6|89.8|

## Difference from the original paper

There are several differences from original paper.
These difference might be the reason for lower accuracy compared to the original paper.

1. In the original paper batch normalization is applied within a single graph whereas our implementaion
   apply batch normalization across multiple graphs.
2. Number of constants: we used 2753 unique tokens + three special tokens "VAR", "VARFUNC", "UNKNOWN",
   whereas the original paper uses only 1906 + 3 tokens.
   We used only limited normalization of tokens, but the original paper might used more normilization.

## References

* [1] M. Wang, Y. Tang, J. Wang, and J. Deng, "Premise selection for theorem proving by deep graph embedding," Sep. 2017.
  Available: http://arxiv.org/abs/1709.09994

## License

[MIT License](LICENSE)

`sparse_matmul.py` is copied from https://github.com/pfnet-research/chainer-chemistry/pull/90
