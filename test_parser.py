import itertools
import sys
import parsy
import holstep
import parser

sys.setrecursionlimit(10000)

for fname in itertools.chain(("holstep/train/%05d" % i for i in range(1,10000)), ("holstep/test/%04d" % i for i in range(1,1412))):
    print(fname)
    file = holstep.read_file(fname)
    num_error = 0
    for formula in itertools.chain([file.conjecture], file.dependencies, file.examples):
        try:
            parser.judgement.parse(formula.text)
        except parsy.ParseError as ex:
            print(ex)
            print(formula.text)
            num_error += 1
            if num_error >= 10:
                break
