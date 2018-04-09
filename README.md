# chainer-formulanet

Chainer implementation of FormulaNet [1].

## Requirement

* Python 3
* Chainer
* funcparserlib

## Usage

```
$ wget http://cl-informatik.uibk.ac.at/cek/holstep/holstep.tgz
$ tar zxf holstep.tgz
$ python build_db.py
$ python formulanet_train.py --dataset resullts
```

## References

* [1] M. Wang, Y. Tang, J. Wang, and J. Deng, "Premise selection for theorem proving by deep graph embedding," Sep. 2017.
  Available: http://arxiv.org/abs/1709.09994

## License

[MIT License](LICENSE)
