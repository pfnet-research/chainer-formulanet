# Chainer implementation of Formulanet

This is an experimental Chainer implementation of FormulaNet [1].

Disclaimer: PFN provides no warranty or support for this implementation. Use it at your own risk.

## Requirement

* Python 3
* [Chainer](https://chainer.org/)
* [funcparserlib](https://pypi.python.org/pypi/funcparserlib/)

## Usage

```
$ wget http://cl-informatik.uibk.ac.at/cek/holstep/holstep.tgz
$ tar zxf holstep.tgz
$ python build_db.py
$ python formulanet_train.py --dataset results
```

## Results

### Validation accuracy with different numbers of update steps on conditional premise selection.

Observed accurarcy in our experiment is similar but somewhat lower.

Results reported in the paper ([1], Table 3):

|Number of steps  | 0  | 1  | 2  | 3  | 4  |
|:----------------|:--:|:--:|:--:|:--:|:--:|
|FormulaNet-basic |81.5|89.3|89.8|89.9|90.0|
|FormulaNet       |81.5|90.4|91.0|91.1|90.8|

Results of our experiment:

|Number of steps  | 0  | 1  | 2  | 3  | 4  |
|:----------------|:--:|:--:|:--:|:--:|:--:|
|FormulaNet-basic |74.2|86.5|88.0|87.8|87.7|
|FormulaNet       |74.2|88.4|89.2|88.9|88.7|

## Difference from the original paper

There are several differences from original paper.
They might be the reason for lower accuracy compared to the original paper.

(1) In the original paper batch normalization is applied within a single graph whereas our implementaion
    apply batch normalization across multiple graphs.
(2) Number of constants: we used 2753 unique tokens + three special tokens "VAR", "VARFUNC", "UNKNOWN",
    whereas the original paper uses only 1906 + 3 tokens.

## References

* [1] M. Wang, Y. Tang, J. Wang, and J. Deng, "Premise selection for theorem proving by deep graph embedding," Sep. 2017.
  Available: http://arxiv.org/abs/1709.09994

## License

[MIT License](LICENSE)

`sparse_matmul.py` is copied from https://github.com/pfnet-research/chainer-chemistry/pull/90
