import itertools
import holstep
from funcparserlib.parser import (NoParseError)
import parser_funcparselib

# sys.setrecursionlimit(10000)

for fname in itertools.chain(("holstep/train/%05d" % i for i in range(1, 10000)),
                             ("holstep/test/%04d" % i for i in range(1, 1412))):
    print(fname)
    file = holstep.read_file(fname)
    num_error = 0
    for formula in itertools.chain([file.conjecture], file.dependencies, file.examples):
        try:
            tokens = parser_funcparselib.tokenize(formula.text)
            parser_funcparselib.thm.parse(tokens)
        except NoParseError as ex:
            print(ex)
            print(formula.text)
            print(tokens)
            num_error += 1
            if num_error >= 10:
                break
