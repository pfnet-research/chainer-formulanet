from pathlib import Path
from typing import List, NamedTuple, Union

Formula = NamedTuple(
    "Formula",
    [("name", str),
     ("text", str),
     ("tokens", str),
    ])

DataFile = NamedTuple(
    "DataFile",
    [("conjecture", Formula),
     ("dependencies", List[Formula]),
     ("examples", List[Formula]),
     ("labels", List[bool]),
     ])

def _readline_with_prefix(f, prefix: str) -> str:
    l = f.readline().strip()
    if not l.startswith(prefix):
        raise RuntimeError("\"{}\" does not start with \"{}\"", l, prefix)
    l = l[len(prefix):]
    return l

def read_file(fname: Union[str,Path]) -> DataFile:
    with open(str(fname)) as f:
        conj_name = _readline_with_prefix(f, "N ")
        conj_text = _readline_with_prefix(f, "C ")
        conj_tokens = _readline_with_prefix(f, "T ")
        conj = Formula(conj_name, conj_text, conj_tokens)

        deps = []
        examples = []
        labels = []

        l = f.readline()
        while (l is not None) and len(l) > 0:
            if l.startswith("D "):
                dep_name = l[2:].strip()
                dep_text = _readline_with_prefix(f, "A ")
                dep_tokens = _readline_with_prefix(f, "T ")
                deps.append(Formula(dep_name, dep_text, dep_tokens))
            elif l.startswith("+ "):
                pos_text = l[2:].strip()
                pos_tokens = _readline_with_prefix(f, "T ")
                examples.append(Formula(None, pos_text, pos_tokens))
                labels.append(True)
            elif l.startswith("- "):
                neg_text = l[2:].strip()
                neg_tokens = _readline_with_prefix(f, "T ")
                examples.append(Formula(None, neg_text, neg_tokens))
                labels.append(False)
            else:
                raise RuntimeError("parse error: " + l)
            l = f.readline()

        return DataFile(conj, deps, examples, labels)
