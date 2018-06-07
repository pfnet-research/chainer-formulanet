from typing import Dict, Generic, List, Set, Text, Tuple, TypeVar, Union
from expr import EIdent, EQuantified, EApply, Expr, Thm

L = TypeVar('L')


# We intentionally not use collections.namedtuple because we need node identity.
# Tree = collections.namedtuple("Tree", ["label", "children"])
class Tree(Generic[L]):
    def __init__(self, label: L, children: List["Tree"]) -> None:
        self.label = label
        self.children = children


def thm_to_tree(thm: Thm) -> Tree[Union[Text, Tuple[Text, Text]]]:
    premises = Tree(",", [expr_to_tree(e) for e in thm.premises])
    conclusion = expr_to_tree(thm.conclusion)
    return Tree("|-", [premises, conclusion])


def expr_to_tree(e: Expr) -> Tree[Union[Text, Tuple[Text, Text]]]:
    if isinstance(e, EIdent):
        return Tree(e.name, [])
    elif isinstance(e, EQuantified):
        return Tree((e.binder, e.var), [expr_to_tree(e.body)])
    elif isinstance(e, EApply):
        args = []
        while isinstance(e, EApply):
            args.append(expr_to_tree(e.arg))
            e = e.fun
        args = list(reversed(args))
        if isinstance(e, EIdent):
            return Tree(e.name, args)
        elif isinstance(e, EQuantified):
            # !!!: ラムダ式の場合、本体と引数が両方ある
            return Tree((e.binder, e.var), [expr_to_tree(e.body)] + args)
        else:
            assert False
    else:
        assert False


def collect_labels(t: Tree[L]) -> Set[L]:
    ret = set()

    def f(t: Tree):
        ret.add(t.label)
        for ch in t.children:
            f(ch)

    f(t)
    return ret


def tree_to_graph(t: Tree[Union[Text, Tuple[Text, Text]]])\
        -> Tuple[List[Text], List[Tuple[int, int]], List[Tuple[int, int, int]]]:
    nodes = []  # type: List[Text]
    edges = []  # type: List[Tuple[int, int]]
    treelets = []  # type: List[Tuple[int, int, int]]

    def new_node(label: Text) -> int:
        n = len(nodes)
        nodes.append(label)
        return n

    def process(t: Tree[Union[Text, Tuple[Text, Text]]], env: Dict[Text, Tuple[int, int]]) -> int:
        n_children = []  # type: List[int]

        if isinstance(t.label, tuple):
            # 変数が実際には使われていなくてもノードを作ってしまっているのは元論文と違うかも
            binder_name, var = t.label
            n = new_node(binder_name)
            n_var = new_node("VAR")
            edges.append((n, n_var))
            env2 = dict(env)
            env2[var] = (n, n_var)
            edges.append((n, process(t.children[0], env2)))
            # 上のfの呼び出しの中で子が追加される可能性があるので、ここで子を集める
            n_children = [n2 for (n1, n2) in edges if n1 == n]
            n_children += process_args(n, t.children[1:], env)
            build_treelets(n, n_children)
            return n
        elif t.label in env:
            binder, n = env[t.label]
            if len(t.children) == 0:
                return n
            else:
                n = new_node("VARFUNC")
                edges.append((binder, n))
                n_children = process_args(n, t.children, env)
                build_treelets(n, n_children)
                return n
        else:
            n = new_node(t.label)
            n_children = process_args(n, t.children, env)
            build_treelets(n, n_children)
            return n

    def process_args(n: int, args, env: Dict[Text, Tuple[int, int]]) -> List[int]:
        n_args = []
        for arg in args:
            n_arg = process(arg, env)
            edges.append((n, n_arg))
            n_args.append(n_arg)
        return n_args

    def build_treelets(n: int, n_children: List[int]) -> None:
        for i, n_ch1 in enumerate(n_children):
            for n_ch2 in n_children[i + 1:]:
                treelets.append((n_ch1, n, n_ch2))

    process(t, {})
    return nodes, edges, treelets


if __name__ == '__main__':
    import holstep
    import parser_funcparselib

    df = holstep.read_file("holstep/train/00001")
    thm = parser_funcparselib.thm.parse(parser_funcparselib.tokenize(df.conjecture.text))
    print(thm)
    t = thm_to_tree(thm)
    print(t)
    print(collect_labels(t))
    g = tree_to_graph(t)
    print(g)
